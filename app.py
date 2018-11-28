# -*- coding: utf-8 -*-
import os
import cv2
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from transformers import FaceSwapper
from style_transfer.src.transformer import StyleTransferWithSegmentationModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

face_swapper = FaceSwapper()
style_transfer = StyleTransferWithSegmentationModel()


def make_style_transfer(filename):
    DESTINATION = 'images/style_transfer'
    os.makedirs(DESTINATION, exist_ok=True)
    destination_filename = os.path.join(DESTINATION, os.path.basename(filename))

    # img = cv2.imread(filename)
    # assert img is not None, filename
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, _ = style_transfer.alter_cloths(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(destination_filename, img)

    img, _ = style_transfer.alter_background(destination_filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(destination_filename, img)

    return destination_filename


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data, folder='./images/upload')
        init_file_url = photos.url(filename)

        mod_fn = face_swapper.transform(filename)
        mod_fn = make_style_transfer(mod_fn)

        mod_file_url = photos.url(mod_fn)
    else:
        init_file_url = None
        mod_file_url = None
    return render_template('index.html', form=form, initial_file_url=init_file_url,
                           modified_file_url=mod_file_url)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # print(make_style_transfer('images/upload/DSC_0696_4.jpg'))
