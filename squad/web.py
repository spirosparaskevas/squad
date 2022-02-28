import pytorch_lightning as pl
import torch

from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix

from squad.constants import CHECKPOINT_PATH
from squad.dataloading import create_inference_dataloader
from squad.qa_model import QAModel

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='SQuAD API')

ns = api.namespace('squad')

request_model = api.model('Request', {
    'context': fields.String(required=True, description='The context'),
    'question': fields.String(required=True, description='The question')
})

response_model = api.model('Response', {
    'answer': fields.String(required=True, description='The predicted answer')
})

model = QAModel.load_from_checkpoint(CHECKPOINT_PATH)
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices="auto",
    precision=16 if torch.cuda.is_available() else 32,
    enable_progress_bar=False,
    enable_model_summary=False
)


@ns.route('/predict')
class QAPrediction(Resource):
    @ns.doc('Predict answer')
    @ns.expect(request_model)
    @ns.marshal_with(response_model)
    def post(self):
        pred_data = [api.payload]
        dataloader = create_inference_dataloader(pred_data, model.tokenizer, batch_size=1)
        predictions = trainer.predict(model, dataloader)
        return {
            'answer': predictions[0][0]
        }
