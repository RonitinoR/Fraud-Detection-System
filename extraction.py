import os
import zipfile
import subprocess

os.environ['Kaggle_config'] = os.path.expanduser('~/.kaggle')

if os.path.exists("creditcardfraud.zip"): raise Exception("The file/s are/is already downloaded!")

#download the dataset
else:
    subprocess.run(["kaggle","datasets", "download", "-d", "mlg-ulb/creditcardfraud"])

    with zipfile.ZipFile("creditcardfraud.zip", 'r') as zip_ref:
        zip_ref.extractall("transaction_data")
    print("Downloaded and extracted the required datasets.")
