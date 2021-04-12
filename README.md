The goal of this project is to detect activities and states using a webcam.
What I am looking to get is an end-to-end solution, which will tell me when I look tired, distructed and in need of a break.

This project is at its early development stage. My rough plan:
- [+] Extract cropped image of a face from a video stream, store them to SQLite database.
- [+] Train VAE to get independent lower-dimensional embeddings of face images.
- [-] Analyse the embeddings, look at their temporal dynamics.
- [-] Make CLIs for data collection and model training, improve usability.
- [-] Provide mechanisms to annotate time frames.
- [-] Create a rapidly trainable classifier based on time frame annotations.
- [-] Combine all the pieces into one coherent whole with a clear workflow.
