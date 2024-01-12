mapboxgl.accessToken = 'pk.eyJ1IjoibWZ1ZW50ZXNhZ3VpbGFyIiwiYSI6ImNsb3NpeXduZzAweHkyaWxsazEzNDFtZmIifQ.7JDeB6ht_cYYH6uPzEBC9Q';

    let map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/streets-v11',
    center: [4.38017, 51.98595],
    zoom: 9
    });

    map.on('click', function (e) {
        var coordinates = e.lngLat;                        
        document.getElementById('id_latitud').value = coordinates.lat;
        document.getElementById('id_longitud').value = coordinates.lng; 
        document.getElementById('id_nombre').value = "Delfx";  
        // Hacer solicitud a la API de Geocodificación
        var query = 'https://api.mapbox.com/geocoding/v5/mapbox.places/' + coordinates.lng + ',' + coordinates.lat + '.json?access_token=pk.eyJ1IjoibWZ1ZW50ZXNhZ3VpbGFyIiwiYSI6ImNsb3NpeXduZzAweHkyaWxsazEzNDFtZmIifQ.7JDeB6ht_cYYH6uPzEBC9Q';

        fetch(query)
            .then(response => response.json())
            .then(data => {
            var locationName = data.features[0].place_name;
            console.log('Nombre de la ubicación:', locationName);
            document.getElementById('id_nombre').value = locationName;
            
        });
        document.getElementById('id_nombre').value = locationName;
        });
        
    