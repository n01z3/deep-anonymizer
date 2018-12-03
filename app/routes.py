# -*- coding: utf-8 -*-
import json
import yaml

import flask
from flask import render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, BooleanField

import cv2

from . import app
from .wrappers import Anonymizer, AnonymizerActions

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

with open('app/paths.yml', 'r') as f:
    paths_config = yaml.load(f)

anonymizer = Anonymizer(paths_config)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField('Upload')


class EditingForm(FlaskForm):
    # photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])

    # if_do_face_swap = BooleanField(label='Face Swap')
    # if_do_cloths_style_transfer = BooleanField(label='Cloths Style Transfer')
    # if_do_background_style_transfer = BooleanField(label='Background Style Transfer')

    pants_to_jeans = SubmitField(u'Pants to Jeans')
    pants_to_skin = SubmitField(u'Pants to Crocodile Skin')

    coat_to_jeans = SubmitField(u'Coat to Jeans')
    coat_to_skin = SubmitField(u'Coat to Crocodile Skin')

    colorize_shoes = SubmitField(u'Colorize Shoes')
    colorize_dress = SubmitField(u'Colorize Dress')

    return_to_upload = SubmitField(u'Return to Upload')

    def get_deep_anonymizer_action(self):
        if self.pants_to_jeans.data:
            return AnonymizerActions.PANTS_TO_JEANS

        if self.pants_to_skin.data:
            return AnonymizerActions.PANTS_TO_SKIN

        if self.coat_to_jeans.data:
            return AnonymizerActions.COAT_TO_JEANS

        if self.coat_to_skin.data:
            return AnonymizerActions.COAT_TO_SKIN

        if self.colorize_shoes.data:
            return AnonymizerActions.COLORIZE_SHOES

        if self.colorize_dress.data:
            return AnonymizerActions.COLORIZE_DRESS

        if self.return_to_upload.data:
            return None

        assert False


def resize_input(path):
    image = cv2.imread(path)
    assert image is not None, path

    h, w = image.shape[:2]

    factor = 1280. / max(h, w)

    new_h = int(h * factor)
    new_w = int(w * factor)

    image = cv2.resize(image, (new_w, new_h))
    cv2.imwrite(path, image)
    return path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        init_image_path = photos.save(form.photo.data, folder='./images/upload')

        init_image_path = resize_input(init_image_path)
        # init_image_url = photos.url(init_image_path)

        main_image_path = anonymizer.anonymize(init_image_path, action=AnonymizerActions.ALL_IN)
        main_image_url = photos.url(main_image_path)

        bottom_image_path = anonymizer.compose_bottom(image_path=init_image_path)
        bottom_image_url = photos.url(bottom_image_path)

    else:
        main_image_url = None
        bottom_image_url = None
    return render_template('index.html', form=form, main_image_url=main_image_url,
                           bottom_image_url=bottom_image_url)


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    form = EditingForm()
    init_image_path = json.loads(flask.request.args['messages'])['path']
    init_image_url = photos.url(init_image_path)
    if form.validate_on_submit():
        # init_image_path = photos.save(form.photo.data, folder='./images/upload')
        if form.return_to_upload.data:
            return flask.redirect(flask.url_for('upload_file'))

        anon_image_path = anonymizer.anonymize(image_path=init_image_path,
                                               action=form.get_deep_anonymizer_action())
        anon_image_url = photos.url(anon_image_path)
    else:
        anon_image_url = None
    return render_template('edit.html', form=form, initial_file_url=init_image_url,
                           modified_file_url=anon_image_url)
