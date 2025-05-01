import gradio as gr
import cv2
import numpy as np

from app.webcam_processor import WebcamProcessor

class AgePredictionApp:    
    def __init__(self):
        self.processor = WebcamProcessor()
        self.effects = ["age_prediction"]

    def process_frame(self, img: np.ndarray, effect: str = "age_prediction"):
        if img is None:
            return np.zeros((128, 128, 3), dtype=np.uint8)
            
        img = np.array(img, dtype=np.uint8).copy()
        
        processed_img = self.processor.process_frame(img, effect)
        
        if effect == "age_prediction":
            if len(processed_img.shape) == 2:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                
        
        return processed_img

    def create_interface(self):
        with gr.Blocks(title="Predição de Idade em Tempo Real") as interface:
            gr.Markdown("# Predição de Idade em Tempo Real")
            gr.Markdown("Habilite a webcam para começar")

            with gr.Row():
                with gr.Column():
                    webcam = gr.Image(
                        sources="webcam",
                        streaming=True,
                        label="Webcam Feed",
                        type="numpy"
                    )
                
                with gr.Column():
                    output = gr.Image(
                        label="Processed Output",
                        type="numpy"
                    )

            webcam.stream(
                fn=self.process_frame,
                inputs=[webcam],
                outputs=output,
                show_progress=True
            )

        return interface

def main():
    try:
        app = AgePredictionApp()
        interface = app.create_interface()
        interface.launch(
            share=True,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        raise

if __name__ == "__main__":
    main()