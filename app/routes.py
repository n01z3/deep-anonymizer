# -*- coding: utf-8 -*-
import yaml

from flask import render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from . import app
from .backend import Transformer

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

with open('app/paths.yml', 'r') as f:
    paths_config = yaml.load(f)

transformer = Transformer(paths_config)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        uploaded_image_path = photos.save(form.photo.data, folder='./images/upload')
        uploaded_image_url = photos.url(uploaded_image_path)

        transformed_image_path = transformer.transform(image_path=uploaded_image_path)
        transformed_image_url = photos.url(transformed_image_path)

    else:
        uploaded_image_url = None
        transformed_image_url = None
    return render_template('index.html', form=form,
                           uploaded_image_url=uploaded_image_url,
                           transformed_image_url=transformed_image_url)
