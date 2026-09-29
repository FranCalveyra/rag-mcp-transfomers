"""Nombres y descripciones de las seis herramientas. Las usan las tools de LangChain (parte 2) y el servidor
MCP (parte 3), así los dos agentes ven exactamente el mismo texto y la comparación es justa.

El modelo decide qué herramienta llamar leyendo estas descripciones: cada una dice qué fuente consulta, para
qué tipo de pregunta sirve y cuáles son los valores válidos del parámetro.
"""

BUSCAR_DOCUMENTOS = (
    "Busca en los documentos oficiales del Hospital Provincial Arroyo Claro: normas y procedimientos que no "
    "cambian de un día a otro. Sirve para horarios fijos (visitas, altas, farmacia, laboratorio, vacunatorio, "
    "donación de sangre), quién puede visitar o quedarse con un paciente, preparación para estudios y cirugías "
    "(ayunos, dietas), requisitos y documentación para turnos, internación, farmacia y coberturas, triage de "
    "la guardia, derechos del paciente, accesos al edificio, control de infecciones, maternidad, pediatría, "
    "kinesiología, salud mental y telemedicina. NO tiene el estado de hoy (camas libres, profesionales de "
    "guardia, turnos disponibles, stock de farmacia ni tiempos de espera actuales). "
    "consulta: el tema a buscar, en español y con las mismas palabras que usó el paciente (por ejemplo "
    "'cómo es una consulta por telemedicina'). Si la pregunta toca varios temas de los documentos, hacé una "
    "búsqueda por tema. Devuelve los fragmentos más relevantes con su documento y sección de origen."
)

CONSULTAR_CAMAS = (
    "Estado de HOY de las camas de internación de un sector: total, ocupadas y libres. Usala para saber si hay "
    "lugar o camas disponibles para internar. "
    "sector: uno de clinica_medica, cirugia_general, terapia_intensiva, pediatria, neonatologia, maternidad."
)

CONSULTAR_GUARDIA = (
    "Profesionales que están de guardia HOY en una especialidad, con su horario (por ejemplo 08:00-20:00 es "
    "el turno de día y 20:00-08:00 el de la noche). Usala para saber quién atiende hoy o esta noche. "
    "especialidad: una de clinica_medica, cardiologia, pediatria, traumatologia, obstetricia, salud_mental."
)

CONSULTAR_TURNOS = (
    "Próximos turnos disponibles (fecha y hora) con una especialidad en consultorios externos. Usala para saber "
    "cuándo es el primer turno libre. No dice qué hay que llevar al turno: eso está en los documentos. "
    "especialidad: una de cardiologia, dermatologia, traumatologia, psicologia, gastroenterologia, "
    "obstetricia, kinesiologia."
)

CONSULTAR_FARMACIA = (
    "Stock de HOY de un medicamento en la farmacia del hospital y, si no hay, la fecha de reposición. Usala para "
    "saber si un medicamento está disponible. No dice cómo retirarlo ni qué papeles hacen falta: eso está en los "
    "documentos. "
    "medicamento: uno de 'amoxicilina 500 mg', 'enalapril 10 mg', 'metformina 850 mg', 'salbutamol aerosol', "
    "'insulina NPH', 'paracetamol 500 mg', 'levotiroxina 50 mcg'."
)

CONSULTAR_ESPERA = (
    "Minutos de espera ACTUALES en la guardia para cada nivel de triage (rojo, naranja, amarillo, verde, azul). "
    "Usala para saber cuánto se está esperando hoy en la guardia. No recibe parámetros. Los tiempos máximos "
    "que fija la norma de triage están en los documentos, no acá."
)

DESCRIPCIONES = {
    "buscar_documentos": BUSCAR_DOCUMENTOS,
    "consultar_camas": CONSULTAR_CAMAS,
    "consultar_guardia": CONSULTAR_GUARDIA,
    "consultar_turnos": CONSULTAR_TURNOS,
    "consultar_farmacia": CONSULTAR_FARMACIA,
    "consultar_espera": CONSULTAR_ESPERA,
}
