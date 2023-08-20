import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

vehicle_data = {}

def update_vehicle_table(vehicle_id, color, vehicle_type, model, make_vendor, license_plate, license_plate_color, vehicle_image_path, license_plate_image_path):
    # Open the vehicle image and license plate image using PIL
    try:
        vehicle_image = Image.open(vehicle_image_path)
    except FileNotFoundError:
        print(f'Error: Could not open vehicle image {vehicle_image_path}')
        return

    try:
        license_plate_image = Image.open(license_plate_image_path)
    except FileNotFoundError:
        print(f'Error: Could not open license plate image {license_plate_image_path}')
        return

    # Add the vehicle information to the data dictionary
    vehicle_data[vehicle_id] = {'color': color, 'vehicle_type': vehicle_type, 'model': model, 'make_vendor': make_vendor, 'license_plate': license_plate, 'license_plate_color': license_plate_color, 'vehicle_image': vehicle_image, 'license_plate_image': license_plate_image}

    # Create or update the table
    fig, ax = plt.subplots()
    ax.axis('off')

    table_data = [['Vehicle ID', 'Color', 'Type', 'Model', 'Make Vendor', 'License Plate', 'LP Color', 'Vehicle Image', 'LP Image']]
    for id, data in vehicle_data.items():
        row = [id, data['color'], data['vehicle_type'], data['model'], data['make_vendor'], data['license_plate'], data['license_plate_color'], data['vehicle_image'], data['license_plate_image']]
        table_data.append(row)

    print(table_data) # Check that the vehicle data is being added correctly

    if hasattr(update_vehicle_table, 'table'):
        update_vehicle_table.table.remove()
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.set_fontsize(12)

    # Display the images in the table
    for i in range(1, len(table_data)):
        vehicle_image = table_data[i][7]
        license_plate_image = table_data[i][8]
        vehicle_ax = table._cells[i, 7].get_text()._renderer if table._cells[i, 7].get_text() is not None else None
        license_plate_ax = table._cells[i, 8].get_text()._renderer if table._cells[i, 8].get_text() is not None else None
        if vehicle_ax is not None:
            vehicle_ax.imshow(vehicle_image)
            vehicle_ax.set_aspect('equal')
        if license_plate_ax is not None:
            license_plate_ax.imshow(license_plate_image)
            license_plate_ax.set_aspect('equal')

    update_vehicle_table.table = table

    plt.show()


vehicle_image = Image.open(
    '/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex1/crops/car/6/251.jpg')
vehicle_image.show()

update_vehicle_table(
    vehicle_id=1,
    color='Red',
    vehicle_type='Sedan',
    model='Camry',
    make_vendor='Toyota',
    license_plate='ABC123',
    license_plate_color='Blue',
    vehicle_image_path='/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex1/crops/car/6/251.jpg',
    license_plate_image_path='/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex1/crops/car/6/license_plate.jpg'
)
