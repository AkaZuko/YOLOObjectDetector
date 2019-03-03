BASE_DIR=$(python -c "from os import path; print(path.dirname(path.realpath(\"$0\")))")

echo "Installing required packages"

pip install -r requirements.txt
pip install --upgrade --force-reinstall tensorflow
pip install $BASE_DIR/lib/object_detection-0.1.tar.gz
pip install $BASE_DIR/lib/slim-0.1.tar.gz

echo "Done"
