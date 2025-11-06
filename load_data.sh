# Install AIStor client
curl --progress-bar -L https://dl.min.io/aistor/mc/release/linux-amd64/mc --create-dirs -o $HOME/aistor-binaries/mc
chmod +x ~/aistor-binaries/mc

# Copy data from the bucket
~/aistor-binaries/mc alias set dc24 https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY
~/aistor-binaries/mc cp --recursive dc24/prc-2025-datasets/  ~/prc-challenge-2025/data
