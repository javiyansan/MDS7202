"""Frontend de Gradio para clasificar la prioridad de tickets de ChaucherApp."""

import gradio as gr
from services import enviar_prediccion

# --- Opciones permitidas ---
CANALES = ["Correo", "Página Web", "Whatsapp"]
CATEGORIAS = ["Cobros", "Cuenta", "Fraude", "Otro", "Pregunta general", "Técnica"]
TIPOS_CUENTA = ["Business", "Free", "Premium"]

# Colores asociados a cada prioridad, para que la respuesta se lea de un vistazo.
COLOR_PRIORIDAD = {
    "Baja": "#2e7d32",
    "Media": "#f9a825",
    "Alta": "#ef6c00",
    "Critica": "#c62828",
}

# Paleta acorde a la estética de ChaucherApp (verdes/teal de finanzas + azul del logo).
tema = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
)

CSS = """
#titulo {text-align: center;}
#titulo h1 {color: #1b5e4f; margin-bottom: 0;}
#subtitulo {text-align: center; color: #5f6b6a; margin-top: 0;}
.card {border-radius: 14px; padding: 6px 12px;}
"""


def clasificar(
    asunto,
    contenido,
    canal_ticket,
    categoria_problema,
    tipo_cuenta,
    antiguedad_cuenta_dias,
    fecha_envio,
):
    """Arma el payload, llama al backend y devuelve un HTML con el resultado."""
    if not asunto or not contenido:
        return "<div class='card' style='background:#fdecea;color:#c62828;'>Debes completar el asunto y el contenido del ticket.</div>"

    payload = {
        "asunto": asunto,
        "contenido": contenido,
        "canal_ticket": canal_ticket,
        "categoria_problema": categoria_problema,
        "tipo_cuenta": tipo_cuenta,
        "antiguedad_cuenta_dias": int(antiguedad_cuenta_dias),
        "fecha_envio": str(fecha_envio),
    }

    try:
        resultado = enviar_prediccion(payload)
        prioridad = resultado["nivel_prioridad"]
    except Exception as e:
        return f"<div class='card' style='background:#fdecea;color:#c62828;'>Error al consultar el modelo: {e}</div>"

    color = COLOR_PRIORIDAD.get(prioridad, "#455a64")
    return (
        f"<div class='card' style='background:{color};color:white;text-align:center;'>"
        f"<h2 style='margin:8px 0;'>Prioridad: {prioridad}</h2></div>"
    )


with gr.Blocks(theme=tema, css=CSS, title="ChaucherApp · Priorización de Tickets") as demo:
    gr.Markdown("# ChaucherApp · Priorización de Tickets", elem_id="titulo")
    gr.Markdown(
        "Clasifica automáticamente la prioridad de un ticket de soporte.",
        elem_id="subtitulo",
    )

    with gr.Row():
        # 1.atributos del ticket
        with gr.Column(scale=1):
            gr.Markdown("### Atributos del ticket")
            asunto = gr.Textbox(label="Asunto", placeholder="Ej: Cobro desconocido en mi tarjeta")
            contenido = gr.Textbox(
                label="Contenido",
                placeholder="Describe el problema del ticket...",
                lines=5,
            )
            canal_ticket = gr.Dropdown(CANALES, label="Canal del ticket", value="Whatsapp")
            categoria_problema = gr.Dropdown(CATEGORIAS, label="Categoría del problema", value="Cobros")
            fecha_envio = gr.Textbox(label="Fecha de envío (YYYY-MM-DD)", value="2024-01-02")

        # 2. atributos del usuario
        with gr.Column(scale=1):
            gr.Markdown("### Atributos del usuario")
            tipo_cuenta = gr.Dropdown(TIPOS_CUENTA, label="Tipo de cuenta", value="Free")
            antiguedad_cuenta_dias = gr.Number(
                label="Antigüedad de la cuenta (días)", value=180, minimum=0, precision=0
            )
            gr.Markdown("&nbsp;")
            boton = gr.Button("Clasificar prioridad", variant="primary", size="lg")
            salida = gr.HTML(label="Resultado")

    boton.click(
        fn=clasificar,
        inputs=[
            asunto,
            contenido,
            canal_ticket,
            categoria_problema,
            tipo_cuenta,
            antiguedad_cuenta_dias,
            fecha_envio,
        ],
        outputs=salida,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
