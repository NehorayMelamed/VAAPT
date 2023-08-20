
### IMGAUG: ###
imgaug_parameters = get_IMGAUG_parameters()
# Affine:
imgaug_parameters['flag_affine_transform'] = True
imgaug_parameters['affine_scale'] = 1
imgaug_parameters['affine_translation_percent'] = None
imgaug_parameters['affine_translation_number_of_pixels'] = (0,max_shift_in_pixels)
imgaug_parameters['affine_rotation_degrees'] = 0
imgaug_parameters['affine_shear_degrees'] = 0
imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
imgaug_parameters['probability_of_affine_transform'] = 1
#Perspective:
imgaug_parameters['flag_perspective_transform'] = False
imgaug_parameters['flag_perspective_transform_keep_size'] = True
imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
imgaug_parameters['probability_of_perspective_transform'] = 1
### Get Augmenter: ###
imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)