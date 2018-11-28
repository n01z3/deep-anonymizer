# -*- coding: utf-8 -*-
import yaml

from flask import render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, BooleanField

from . import app
from .wrappers import Anonymizer, AnonymizerActions

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

with open('app/paths.yml', 'r') as f:
    paths_config = yaml.load(f)

anonymizer = Anonymizer(paths_config)


#
# def make_style_transfer(filename):
#     DESTINATION = 'images/style_transfer'
#     os.makedirs(DESTINATION, exist_ok=True)
#     destination_filename = os.path.join(DESTINATION, os.path.basename(filename))
#
#     # img = cv2.imread(filename)
#     # assert img is not None, filename
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     img, _ = style_transfer.alter_cloths(filename)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(destination_filename, img)
#
#     img, _ = style_transfer.alter_background(destination_filename)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(destination_filename, img)
#
#     return destination_filename


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])

    if_do_face_swap = BooleanField(label='Face Swap')
    if_do_cloths_style_transfer = BooleanField(label='Cloths Style Transfer')
    if_do_background_style_transfer = BooleanField(label='Background Style Transfer')

    submit = SubmitField(u'Upload')

    def get_deep_anonymizer_actions(self):
        return {
            AnonymizerActions.FACE_SWAP: self.if_do_face_swap.data,
            AnonymizerActions.CLOTHS_STYLE_TRANSFER: self.if_do_cloths_style_transfer.data,
            AnonymizerActions.BACKGROUND_STYLE_TRANSFER: self.if_do_background_style_transfer.data,
        }


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        init_image_path = photos.save(form.photo.data, folder='./images/upload')
        init_image_url = photos.url(init_image_path)

        anon_image_path = anonymizer.anonymize(image_path=init_image_path,
                                               actions=form.get_deep_anonymizer_actions())
        anon_image_url = photos.url(anon_image_path)
    else:
        init_image_url = None
        anon_image_url = None
    return render_template('index.html', form=form, initial_file_url=init_image_url,
                           modified_file_url=anon_image_url)
