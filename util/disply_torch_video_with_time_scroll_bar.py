import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

def display_video(video_tensor):
    # Convert the tensor to a numpy array
    video_np = video_tensor.permute(0, 2, 3, 1).numpy()

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create a slider for the current frame
    axframe = plt.axes([0.2, 0.01, 0.6, 0.03])
    slider_frame = Slider(axframe, 'Frame', 0, video_np.shape[0] - 1, valinit=0, valstep=1)

    # Create buttons to skip forward and backward
    axprev = plt.axes([0.05, 0.01, 0.1, 0.03])
    axnext = plt.axes([0.85, 0.01, 0.1, 0.03])
    button_prev = Button(axprev, 'Prev')
    button_next = Button(axnext, 'Next')

    # Define a function to update the displayed frame
    def update(frame):
        ax.imshow(video_np[int(frame)])
        fig.canvas.draw_idle()

    # Define a function to handle the button clicks
    def skip(event):
        if event == 'next':
            slider_frame.set_val(slider_frame.val + 1)
        elif event == 'prev':
            slider_frame.set_val(slider_frame.val - 1)

    # Connect the slider and buttons to their respective functions
    slider_frame.on_changed(update)
    button_prev.on_clicked(lambda x: skip('prev'))
    button_next.on_clicked(lambda x: skip('next'))

    # Display the first frame
    ax.imshow(video_np[0])

    # Show the plot
    plt.show()
