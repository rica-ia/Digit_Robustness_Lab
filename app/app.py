"""Start the presentation portal and the original Gradio interface locally."""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
from web import app, ROOT

if __name__ == '__main__':
    import argparse
    import uvicorn
    import gradio as gr
    import gradio_app
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    gradio_app.ROOT = ROOT
    gradio_app.ARTIFACTS = ROOT / 'artifacts'
    app = gr.mount_gradio_app(app, gradio_app.build_demo(), path='/gradio')
    uvicorn.run(app, host='127.0.0.1', port=args.port)
