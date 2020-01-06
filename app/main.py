from fastapi import FastAPI, File, HTTPException
import io
import torch
from PIL import Image
from app import model
from businesslogic.net_utils import process_prediction

app = FastAPI()


@app.post("/predict/")
def predict(file: bytes = File(...)):

    try:
        img = Image.open(io.BytesIO(file)).convert('RGB')
        del file
    except (ValueError, AttributeError, OSError):
        raise HTTPException(status_code=422,
                            detail='You must pass binaries of jpg or png image. Your data can not be interpreted as '
                                   'image')

    with torch.no_grad():
        raw_preds = model(model.transforms(img).unsqueeze(0).to(model.device))

    ingr_ids = raw_preds['ingr_ids'].cpu().numpy()
    recipe_ids = raw_preds['recipe_ids'].cpu().numpy()
    del raw_preds

    outs = process_prediction(recipe_ids[0], ingr_ids[0], model.instr_vocab, model.ingred_vocab)

    return dict(
            title=outs['title'],
            ingredients=outs['ingredients']
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
