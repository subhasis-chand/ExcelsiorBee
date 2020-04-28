import os
from flask import jsonify, request


UPLOAD_DIRECTORY = "./temp/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def upload_input_file(filename):
    for filename in os.listdir('./temp'):
        file_path = os.path.join('./temp', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    file = request.files['file']
    file.save(os.path.join(UPLOAD_DIRECTORY, 'inputFile.csv'))
    s = os.popen('head -10 temp/inputFile.csv').read()
    content = s.split('\n')[:-1]
    content = [x.split(',') for x in content]
    return jsonify({'status': 'success', 'content': content})