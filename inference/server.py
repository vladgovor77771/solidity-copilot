from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch
import sys
from transformers import RobertaTokenizer, T5ForConditionalGeneration

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using', device)


class MyRequestHandler(BaseHTTPRequestHandler):
    tokenizer: RobertaTokenizer
    model: T5ForConditionalGeneration

    def __init__(self, model, tokenizer, *args):
        self.model = model
        self.tokenizer = tokenizer
        super().__init__(*args)

    def do_POST(self):
        if self.path == '/predict':
            # Read the request body and parse as JSON
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            input_text = json.loads(body)['text']

            # Tokenize the input text
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                max_length=512,
                padding="max_length",
                truncation=True
            ).to(device)
            # input_ids[0]
            print(len(input_ids[0]))# input_ids[0])

            # extra_ids = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
            # extra_pos = (input_ids == extra_ids).nonzero()
            # inputs_embeds = self.model.get_input_embeddings()(input_ids)
            # # print(extra_ids, extra_pos, inputs_embeds)
            # inputs_embeds[extra_pos[0, 0], extra_pos[0, 1]] = 1
            # Make inference with the model
            output_ids = self.model.generate(
                input_ids,
                # max_length=128,
                # num_return_sequences=1,
                # do_sample=True,
                # top_k=1,
                # top_p=0.95,
                # temperature=0.9,
                # use_cache=True,
                # inputs_embeds=inputs_embeds
            )
            print(output_ids)

            # Decode the output text
            output_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True)

            # Send the response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'output': output_text})
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)


assert len(sys.argv) == 2, "Args: checkpoint"
checkpoint = sys.argv[1]

# checkpoint = "Salesforce/codet5-large"

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# Create the HTTP server
server_address = ('', 8000)
httpd = HTTPServer(
    server_address, lambda *args: MyRequestHandler(model, tokenizer, *args))
print(f'Starting HTTP server on port {server_address[1]}')
httpd.serve_forever()
