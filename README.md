# calorimeter

## How to run
1. Clone repo
1. Download all weights (see README.txt in weights dir)
1. Go to calorimeter dir
1. Build docker image
```docker build -t calorimeter_im .```
1. Run docker container
```docker run -d --name calorimeter -p 80:80 -e MODULE_NAME="app.main" calorimeter_im```

## API
To get prediction send post request with binaries of an image on ```http://127.0.0.1/predict/```

With Python this will look like
```
url = 'http://127.0.0.1/predict/'
files = {'file':open(impath, 'rb')}
res = requests.post(url, files=files)
```
The result will be a json:
```
{
  "title":"some generated name .",
  "ingredients":["ingred_1","ingred_2","ingred_...","ingred_N"]
}
```
If there is no . in title this means that the model wanted to create a really long name and was stopped not to spend too much time.

For now you can run the model only on CPU
