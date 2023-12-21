from owslib.wps import WebProcessingService
from owslib.wps import ComplexDataInput

def llamar_servicio_pywps(input_parameter):
    # URL del servicio PyWPS
    pywps_url = "http://localhost:5000/wps"
    
    # Crear una instancia de WebProcessingService
    wps = WebProcessingService(pywps_url, version='1.0.0')
    
    # Identificador del proceso PyWPS
    proceso_id = 'mi_proceso_pywps'
    
    # Construir los datos de entrada para el proceso PyWPS
    input_data = ComplexDataInput(input_parameter)
    
    # Construir la solicitud Execute con los parámetros
    execution = wps.execute(process=proceso_id, inputs=[('input_parameter', input_data)], output='output_parameter')

    # Enviar la solicitud Execute
    response = wps.getresponse(execution.request)

    # Verificar si la solicitud fue exitosa (código de estado 200)
    if response.status_code == 200:
        # Extraer los resultados de la respuesta
        resultado = response.read()
        return resultado
    else:
        return f"Error al llamar al servicio PyWPS. Código de estado: {response.status_code}"

