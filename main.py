import io
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
import uvicorn
from models import BERTEmbeddings, ResNetFeatureExtraction
from opensearch.services.opensearch_service import OpensSearchService
import json

app = FastAPI()


class SearchRequest(BaseModel):
    searchType: str
    query: str


class Product(BaseModel):
    productId: int
    name: str
    description: str


text_model = BERTEmbeddings()
image_model = ResNetFeatureExtraction()
opensearch_service = OpensSearchService(
    host="localhost", port=9200, index_name="products")


@app.on_event("startup")
async def startup():
    opensearch_service.delete_index()
    if opensearch_service.is_index_empty():
        with open('data/products.json', 'r') as f:
            products = json.load(f)
            for product in products:
                vectors = text_model(product['name'] +
                                     " : " + product['description'])
                opensearch_service.index_product(
                    product['productId'],
                    vectors,
                    product
                )


@app.on_event("shutdown")
async def shutdown():
    opensearch_service.close()


@app.get("/ping")
def index_item():
    return {"message": "API is UP"}


@app.post("/index-text")
async def index_item(product: Product):
    try:
        vectors = text_model(product.name + " : " + product.description)
        opensearch_service.index_product(
            product.productId, vectors, product.model_dump(), type="text")
        return {"message": "Item indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-image")
async def index_image(file: UploadFile = File(...), productId: int = Form(...), name: str = Form(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensors = image_model(image)
        opensearch_service.index_product(
            productId,
            image_tensors,
            metadata={"name": name},
            type="image"
        )
        return {"message": "Item indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_items(request: SearchRequest):
    if request.searchType not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="Invalid search type")

    vectors = text_model(request.query)
    results = opensearch_service.search_similar_products(
        vectors, top_k=5, type=request.searchType)
    if not results:
        return {"message": "No results found"}

    return results


@app.post("/search-image")
async def search_items(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_tensors = image_model(image)
    results = opensearch_service.search_similar_products(
        image_tensors, top_k=2, type="image")
    if not results:
        return {"message": "No results found"}

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
