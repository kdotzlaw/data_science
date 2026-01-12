import kagglehub


if __name__ == '__main__':
    # Download latest version
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

    print("Path to dataset files:", path)