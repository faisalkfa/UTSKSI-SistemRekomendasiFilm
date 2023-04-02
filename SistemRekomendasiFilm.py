import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Membaca data film
movies = pd.read_csv("movies.csv")

# Membaca data rating
ratings = pd.read_csv("ratings.csv")

# Menggabungkan data film dan rating
data = pd.merge(movies, ratings, on='movieId')

# Membuat pivot table untuk menampilkan rating tiap film
pivot_table = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Mengubah pivot table menjadi array numpy
matrix = pivot_table.to_numpy()

# Menghitung cosine similarity
cosine_sim = cosine_similarity(matrix)

# Membuat dictionary untuk menyimpan index setiap film
indices = pd.Series(movies.index, index=movies['title'])

# Fungsi rekomendasi film
def recommend_movies(title, cosine_sim=cosine_sim, movies=movies):
    # Mencari index dari film yang dipilih
    idx = indices[title]

    # Mencari similarity score dari film yang dipilih
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Mengurutkan film berdasarkan similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mengambil index dari 10 film terbaik
    sim_scores = sim_scores[1:11]

    # Mengambil index film
    movie_indices = [i[0] for i in sim_scores]

    # Menampilkan 10 rekomendasi film
    recommended_movies = movies[['title', 'genres']].iloc[movie_indices]
    recommended_movies['score'] = [i[1] for i in sim_scores]
    recommended_movies = recommended_movies.reset_index(drop=True)
    recommended_movies.index += 1

    # Menambahkan nama kolom
    result = "No | Title | Genres | Score\n"
    result += "-"*80 + "\n"  # Tambahkan garis pemisah

    # Mengubah output ke dalam format yang diinginkan
    for index, row in recommended_movies.iterrows():
        result += f"{index} | {row['title']} | {row['genres']} | {row['score']:.2f}\n"

    return result.strip()

# Meminta pengguna memasukkan judul film
query = input("Masukkan judul film yang ingin direkomendasikan: ")

# Mencari semua film yang memiliki substring yang diinginkan
matched_movies = movies[movies['title'].str.contains(query, case=False)]

# Memeriksa apakah ada film yang cocok
if len(matched_movies) == 0:
    print("Tidak ada film yang cocok dengan kriteria pencarian Anda.")
else:
    # Menampilkan daftar film yang cocok
    print("Daftar film yang cocok:")
    for title in matched_movies['title']:
        print(title)

    # Meminta pengguna untuk memilih judul film
    title = input("Masukkan Judul Lengkap (Beserta Tahunnya) yang ingin direkomendasikan: ")

    # Memanggil fungsi rekomendasi dan mencetak hasilnya
    if title not in matched_movies['title'].values:
        print("Judul film tidak valid.")
    else:
        print(recommend_movies(title))
        
# Hasil Evaluasi

# Mean Average Precision at K (MAP@K)
# Mengimport library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load dataset movies dan ratings
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Bagi data menjadi data latih dan data uji
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Fungsi untuk menghitung Average Precision at K (AP@K)
def average_precision_at_k(recommended_movies, relevant_movies, k):
    if len(recommended_movies) == 0:
        return 0.0

    # Menghitung nilai AP@K
    score = 0.0
    num_hits = 0
    for i, movie in enumerate(recommended_movies[:k]):
        if movie in relevant_movies:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / min(len(relevant_movies), k)

# Variabel untuk menyimpan nilai MAP@K
mapk_scores = []
# Jumlah rekomendasi yang diambil
K = 10  

# Looping melalui setiap pengguna dalam data uji
for user_id in test_data['userId'].unique():
    # Memfilter data uji hanya untuk pengguna ini
    user_ratings = test_data[test_data['userId'] == user_id]
    # Memilih film yang telah diberi peringkat oleh pengguna
    user_movies = movies[movies['movieId'].isin(user_ratings['movieId'])]
    # Mendapatkan genre dari film yang telah diberi peringkat oleh pengguna
    user_genres = user_movies['genres'].tolist()
    
    # Membuat set dari film yang telah diberi peringkat oleh pengguna
    relevant_movies = set(user_movies['title'].tolist())

    # Looping melalui setiap genre film yang disukai oleh pengguna
    for movie_genres in user_genres:
        # Menghitung skor kesamaan antara setiap film dengan film yang dipilih berdasarkan genre
        sim_scores = []
        for idx, movie in movies.iterrows():
            score = len(set(movie_genres.split('|')).intersection(set(movie['genres'].split('|'))))
            sim_scores.append((idx, score))
        
        # Mengurutkan film berdasarkan skor kesamaan dengan urutan menurun dan ambil 10 film teratas
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:K+1]

        # Mendapatkan daftar film yang direkomendasikan
        recommended_movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies.iloc[recommended_movie_indices]['title'].tolist()

        # Menghitung skor Average Precision at K (AP@K) dan menambahkannya ke dalam variabel mapk_scores
        mapk_score = average_precision_at_k(recommended_movies, relevant_movies, K)
        mapk_scores.append(mapk_score)

# Menghitung Mean Average Precision at K (MAP@K) dari semua nilai AP@K yang telah dihitung
mean_average_precision_at_k = np.mean(mapk_scores)
# Menampilkan hasil Mean Average Precision at K (MAP@K)
print(f"Mean Average Precision at K: {mean_average_precision_at_k:.4f}")

# Mean Reciprocal Rank
# Fungsi untuk menghitung nilai Reciprocal Rank (RR)
def reciprocal_rank(recommended_movies, relevant_movies):
    # Looping pada setiap film dalam daftar rekomendasi
    for i, movie in enumerate(recommended_movies):
        # Jika film relevan ditemukan, kembalikan nilai RR
        if movie in relevant_movies:
            return 1 / (i + 1)
    # Jika tidak ditemukan, kembalikan nilai 0
    return 0

# Variabel untuk menyimpan semua nilai RR
rr_scores = []

# Looping pada setiap pengguna dalam data uji
for user_id in test_data['userId'].unique():
    # Memfilter data uji hanya untuk pengguna ini
    user_ratings = test_data[test_data['userId'] == user_id]
    # Memilih film yang telah diberi peringkat oleh pengguna
    user_movies = movies[movies['movieId'].isin(user_ratings['movieId'])]
    # Mendapatkan genre dari film yang telah diberi peringkat oleh pengguna
    user_genres = user_movies['genres'].tolist()
    
    # Membuat set dari film yang telah diberi peringkat oleh pengguna
    relevant_movies = set(user_movies['title'].tolist())

    # Looping pada setiap genre film yang disukai oleh pengguna
    for movie_genres in user_genres:
        # Menghitung skor kesamaan antara setiap film dengan film yang dipilih berdasarkan genre
        sim_scores = []
        for idx, movie in movies.iterrows():
            score = len(set(movie_genres.split('|')).intersection(set(movie['genres'].split('|'))))
            sim_scores.append((idx, score))
        
        # Mengurutkan film berdasarkan skor kesamaan dengan urutan menurun dan ambil 10 film teratas
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:K+1]

        # Mendapatkan daftar film yang direkomendasikan
        recommended_movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies.iloc[recommended_movie_indices]['title'].tolist()

        # Menghitung nilai RR dan menambahkannya ke dalam variabel rr_scores
        rr_score = reciprocal_rank(recommended_movies, relevant_movies)
        rr_scores.append(rr_score)

# Menghitung Mean Reciprocal Rank (MRR) dari semua nilai RR yang telah dihitung
mean_reciprocal_rank = np.mean(rr_scores)
# Menampilkan hasil Mean Reciprocal Rank (MRR)
print(f"Mean Reciprocal Rank: {mean_reciprocal_rank:.4f}")

# Coverage
# Variabel untuk menyimpan semua film yang direkomendasikan oleh sistem
unique_recommended_movies = set()

# Looping pada setiap pengguna dalam data uji
for user_id in test_data['userId'].unique():
    # Ambil film-film yang telah diberi peringkat oleh pengguna
    user_ratings = test_data[test_data['userId'] == user_id]
    # Ambil film dari dataset movies yang telah diberi peringkat oleh pengguna
    user_movies = movies[movies['movieId'].isin(user_ratings['movieId'])]
    # Ambil genre dari film-film yang telah diberi peringkat oleh pengguna
    user_genres = user_movies['genres'].tolist()
    
    # Buat set dari film-film yang telah diberi peringkat oleh pengguna
    relevant_movies = set(user_movies['title'].tolist())

    # Looping pada setiap genre film
    for movie_genres in user_genres:
        # Hitung skor kesamaan antara setiap film dengan film yang dipilih berdasarkan genre
        sim_scores = []
        for idx, movie in movies.iterrows():
            score = len(set(movie_genres.split('|')).intersection(set(movie['genres'].split('|'))))
            sim_scores.append((idx, score))
        
        # Urutkan film-film yang memiliki skor kesamaan dengan urutan menurun dan ambil 10 film teratas
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:K+1]

        # Ambil indeks film-film yang direkomendasikan
        recommended_movie_indices = [i[0] for i in sim_scores]
        # Ambil judul film-film yang direkomendasikan dan simpan di dalam variabel unique_recommended_movies
        recommended_movies = movies.iloc[recommended_movie_indices]['title'].tolist()
        unique_recommended_movies.update(recommended_movies)

# Menghitung nilai Coverage dari sistem rekomendasi
coverage = len(unique_recommended_movies) / len(movies)
# Menampilkan hasil nilai Coverage
print(f"Coverage: {coverage:.4f}")

# Menghitung presisi, recall, dan f-measure dari daftar film yang direkomendasikan
# dengan membandingkannya dengan daftar film yang relevan yang diketahui oleh pengguna
def precision_recall_fmeasure_at_k(recommended_movies, relevant_movies, k):
    if len(recommended_movies) == 0:
        return 0.0, 0.0, 0.0

    # Menghitung jumlah film yang direkomendasikan yang juga relevan dengan pengguna
    num_hits = 0
    for i, movie in enumerate(recommended_movies[:k]):
        if movie in relevant_movies:
            num_hits += 1

    # Menghitung presisi dan recall
    precision = num_hits / k
    recall = num_hits / len(relevant_movies)
    
    # Menghitung f-measure
    if precision + recall == 0:
        f_measure = 0.0
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    return precision, recall, f_measure

# Inisialisasi array untuk menyimpan skor presisi, recall, dan f-measure
precision_scores = []
recall_scores = []
f_measure_scores = []

# Looping melalui setiap pengguna di data uji
for user_id in test_data['userId'].unique():
    # Memfilter data uji hanya untuk pengguna ini
    user_ratings = test_data[test_data['userId'] == user_id]
    user_movies = movies[movies['movieId'].isin(user_ratings['movieId'])]
    user_genres = user_movies['genres'].tolist()

    # Membuat daftar film yang relevan untuk pengguna ini
    relevant_movies = set(user_movies['title'].tolist())

    # Looping melalui setiap genre film yang disukai oleh pengguna
    for movie_genres in user_genres:
        # Menghitung skor kesamaan dengan setiap film dalam database
        sim_scores = []
        for idx, movie in movies.iterrows():
            score = len(set(movie_genres.split('|')).intersection(set(movie['genres'].split('|'))))
            sim_scores.append((idx, score))

        # Mengurutkan film berdasarkan skor kesamaan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:K+1]

        # Mendapatkan daftar film yang direkomendasikan
        recommended_movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies.iloc[recommended_movie_indices]['title'].tolist()

        # Menghitung skor presisi, recall, dan f-measure
        precision, recall, f_measure = precision_recall_fmeasure_at_k(recommended_movies, relevant_movies, K)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f_measure_scores.append(f_measure)

# Menghitung rata-rata skor presisi, recall, dan f-measure untuk semua pengguna
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f_measure = np.mean(f_measure_scores)

# Menampilkan skor presisi, recall, dan f-measure rata-rata
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F-measure: {mean_f_measure:.4f}")
