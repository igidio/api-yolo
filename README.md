# API (YOLOv11 e-commerce)

### Requisitos
* Python (3.10 o superior)

### Instalación
1. Clona el repositorio:
````shell
git clone https://github.com/igidio/yolo-ecommerce.git
````

2. Instalación de dependencias:
```shell
pip install "fastapi[standard]" ultralytics
```
Esto instalará las bibliotecas FastAPI (con características estándar) y Ultralytics, que son fundamentales para el funcionamiento del proyecto.


### Ejecución
Para iniciar el servidor de desarrollo:
```shell
uvicorn main:app --reload --port 8080
```
Donde podemos indicar el puerto si hace falta