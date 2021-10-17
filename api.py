import requests


class Hack4MedAPI:
    HACK4MED_API_URL = 'http://hack4med.eu.ngrok.io/'
    VALIDATION_ENDPOINT = HACK4MED_API_URL + 'file/upload/'

    def __init__(self, name):
        self.name = name
        self.token = name

    def validate_model(self, source_file, model_file):
        with open(source_file, 'rb') as source_fh, open(model_file, 'rb') as model_fh:
            files = {'source_file': source_fh, 'model_file': model_fh}
            data = {'user_token': self.token, 'user_name': self.name}
            response = requests.post(self.VALIDATION_ENDPOINT, files=files, data=data)
            return response

    def results(self):
        response = requests.get(self.VALIDATION_ENDPOINT + self.token)
        return response.json()

    def last_result(self):
        try:
            return self.results()[0]
        except IndexError:
            return None
