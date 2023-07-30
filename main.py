from transformers import pipeline
from flask import Flask, render_template, request
import random
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# モデルとトークナイザーの読み込み
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
classifier = pipeline("sentiment-analysis")

# Spotify APIの設定
client_id = '16d1099bbf9b4361bb7c9946ca5814c9'
client_secret = '5af07dce05744c308c4cdba977c750ca'
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(
    client_credentials_manager=client_credentials_manager, language='ja')


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    return generated_caption


def search_japanese_track_by_danceability_range(data):
    if not data or 'label' not in data[0] or 'score' not in data[0]:
        return "Invalid data format. Please provide data in the correct format."

    mood_label = data[0]['label']
    mood_score = data[0]['score']

    danceability_inf = 0.0
    danceability_sup = 0.5
    energy_inf = 0.0
    energy_sup = 0.5
    valence_inf = 0.0
    valence_sup = 0.5
    mode = 0

    if mood_label == 'POSITIVE':
        # positiveの場合はdanceability、energy、valenceが0.66より大きく1以下、modeが1
        danceability_inf = energy_inf = valence_inf = 0.5
        danceability_sup = energy_sup = valence_sup = 1
        mode = 1
    else:
        mode = 0  # Assign mode 0 for 'NEGATIVE'

    # 検索キーワードを作成
    search_keyword = "TopSongsJapan"

    # Spotify APIを使って日本語の曲を検索
    results = sp.search(q=search_keyword, type='track', limit=50)

    tracks = []
    if results['tracks']['items']:
        for track_info in results['tracks']['items']:
            track_name = track_info['name']
            track_artist = track_info['artists'][0]['name']
            track_url = track_info['external_urls']['spotify']
            track_id = track_info['id']

            # spotipyを使用してトラックの詳細情報を取得
            track_audio_features = sp.audio_features(track_id)
            if track_audio_features:
                track_danceability = track_audio_features[0]['danceability']
                track_energy = track_audio_features[0]['energy']
                track_valence = track_audio_features[0]['valence']
                track_mode = track_audio_features[0]['mode']

                # 各項目が条件に当てはまる曲を抽出
                if valence_inf <= track_valence <= valence_sup and energy_inf <= track_energy <= energy_sup:
                    if mode is not None:
                        if track_mode == mode:
                            tracks.append({
                                'name': track_name,
                                'artist': track_artist,
                                'url': track_url,
                                'danceability': track_danceability,
                                'energy': track_energy,
                                'valence': track_valence,
                                # Display "POSITIVE" if mode is 1, otherwise "NEGATIVE"
                                'mode': "POSITIVE" if mode == 1 else "NEGATIVE"
                            })
                    else:
                        tracks.append({
                            'name': track_name,
                            'artist': track_artist,
                            'url': track_url,
                            'danceability': track_danceability,
                            'energy': track_energy,
                            'valence': track_valence,
                            # Display "POSITIVE" if mode is 1, otherwise "NEGATIVE"
                            'mode': "POSITIVE" if mode == 1 else "NEGATIVE"
                        })

    if not tracks:
        return "NO MUSIC"

    return tracks


#@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = f"/tmp/{image_file.filename}"
            image_file.save(image_path)
            generated_caption = generate_caption(image_path)
            sentiment_data = classifier(generated_caption)
            japanese_tracks = search_japanese_track_by_danceability_range(
                sentiment_data)

            if not japanese_tracks:
                return render_template('no_music.html', caption=generated_caption)

            return render_template('result.html', caption=generated_caption, track=random.choice(japanese_tracks))

    return render_template('index.html')



def showLoading():
    return '<script>document.getElementById("loading").style.display = "block";</script>'


if __name__ == '__main__':
    app.run(debug=True)
