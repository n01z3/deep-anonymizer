import torch
import tensorflow

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
