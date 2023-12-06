# pywps_service.py
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/pywps', methods=['POST'])
def pywps_service():
    # Procesa los datos recibidos desde el formulario Django
    # Puedes acceder a los datos con request.form
    campo1 = request.form.get('ID')
    campo2 = request.form.get('description')
    print("Línea 11")
    campo3 = campo1 + campo2
    print(campo3)

    # Realiza las operaciones necesarias con los datos
    # ...

    # Retorna la respuesta (puedes usar jsonify si es necesario)
    return jsonify({'resultado': campo3})

if __name__ == '__main__':
    app.run(debug=True)
