from owslib.wps import WebProcessingService
from owslib.wps import ComplexDataInput

def llamar_servicio_pywps(input_parameter):
    # URL del servicio PyWPS
    pywps_url = "http://localhost:5000/wps"
    
    # Crear una instancia de WebProcessingService con el proceso deseado
    proceso_id = 'MyWPSProcesss'
    wps = WebProcessingService(pywps_url, version='1.0.0', processid=proceso_id)
    
    # Construir los datos de entrada para el proceso PyWPS
    input_data = ComplexDataInput(input_parameter)
    print("Deltares_Gpp línea 16")
    # Construir la solicitud Execute con los parámetros
    execution = wps.execute(inputs=[('input_parameter', input_data)], output='output_parameter')
    

    print("Deltares_Gpp línea 19")
    # Enviar la solicitud Execute
    response = wps.getresponse(execution.request)

    # Verificar si la solicitud fue exitosa (código de estado 200)
    if response.status_code == 200:
        # Extraer los resultados de la respuesta
        resultado = response.read()
        return resultado
    else:
        return f"Error al llamar al servicio PyWPS. Código de estado: {response.status_code}"

