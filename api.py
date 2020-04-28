import os
from flask_cors import CORS
from flask import Flask, request, abort, jsonify, send_from_directory, url_for, redirect

from src.FileUploads import upload_input_file
from src.EditFile import edit_file

UPLOAD_DIRECTORY = "./temp/"

api = Flask(__name__)
CORS(api)

@api.route('/download/<filename>', methods=["GET", "POST"])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIRECTORY, filename, as_attachment=True)

@api.route("/upload_input_file/<filename>", methods=["POST"])
def UploadInputFile(filename):
    return upload_input_file(filename)

@api.route("/edit_file/", methods=["POST"])
def EditFile():
    deleteCols = request.args.get('deleteCols')
    deleteRows = request.args.get('deleteRows')
    removeHeader = request.args.get('removeHeader')
    removeNaN = request.args.get('removeNaN')
    saveInBE = request.args.get('saveInBE')
    print(deleteCols, deleteRows, removeHeader, removeNaN, saveInBE)
    return edit_file(deleteCols, deleteRows, removeHeader, removeNaN, saveInBE)


if __name__ == "__main__":
    api.run(debug=True, port=5000)