
from __future__ import print_function



# from IIR2Filter import IIR2Filter
# MyFilter = IIR2Filter(order,cutoff,filterType,design='butter',rp=1,rs=1,fs=0)
#
# At the instantiation of the filter the following parameters are compulsory:
#     order:positive integer
#         It represents the order of the desired filter.
#         Can be odd or even number, the filter will create a chain of second
#         order filters and an extra first order one if necessary.
#     cutoff:array/positive float
#         Depending on the desired filter 1 cutoff frequency is to be
#         enetered as a positive float for low/highpass filters or
#         2 cutoff frequenices to be entered in an array as positive floats
#         for bandstop and bandpass filters. These cutoff frequencies can be
#         either entered as normalised to the Nyquist frequency (1 =
#         Nyquist frequency) or as Hz (0 < f < Nyquist), but in this case fs,
#         the sampling frequency has to be passed too.
#     filterType:string
#         Options are: lowpass, highpass, bandpass, bandstop
#
# The following paremeters are NON-compulsory:
#     design:string
#         Different types of coefficient generations
#         can be chosen. The three available filters are Butterworth,
#         Chebyshev type 1 or type 2.
#         The options are: butter, cheby1, cheby2. If left unspecified the
#         default value is butter.
#     rp:positive float
#         Only necessary if cheby1 is chosen to be used. It defines the
#         maximum allowed passband ripples in decibels. If unspecified the
#         default is 1.
#     rs:positive float
#         Only necessary if cheby2 is chosen to be used. It defines the
#         minimum required stopband attenuation in decibels. If unspecified
#         the default is 1.
#     fs:positive float
#         The sampling frequency should only be specified if the cutoff
#         frequency(es) provided are not normalised to Nyquist already.
#         In that case the sampling frequency in Hz will be used to normalise
#         them.




class IIR2Filter(object):

    def createCoeffs(self, order, cutoff, filterType, design='butter', rp=1, rs=1, fs=0):

        # defining the acceptable inputs for the design and filterType params
        self.designs = ['butter', 'cheby1', 'cheby2']
        self.filterTypes1 = ['lowpass', 'highpass', 'Lowpass', 'Highpass', 'low', 'high']
        self.filterTypes2 = ['bandstop', 'bandpass', 'Bandstop', 'Bandpass']

        # Error handling: other errors can arise too, but those are dealt with
        # in the signal package.
        self.isThereAnError = 1  # if there was no error then it will be set to 0
        self.COEFFS = [0]  # with no error this will hold the coefficients

        if design not in self.designs:
            print('Gave wrong filter design! Remember: butter, cheby1, cheby2.')
        elif filterType not in self.filterTypes1 and filterType not in self.filterTypes2:
            print('Gave wrong filter type! Remember: lowpass, highpass', ', bandpass, bandstop.')
        elif fs < 0:
            print('The sampling frequency has to be positive!')
        else:
            self.isThereAnError = 0

        # if fs was given then the given cutoffs need to be normalised to Nyquist
        if fs and self.isThereAnError == 0:
            for i in range(len(cutoff)):
                cutoff[i] = cutoff[i] / fs * 2

        if design == 'butter' and self.isThereAnError == 0:
            self.COEFFS = signal.butter(order, cutoff, filterType, output='sos')
        elif design == 'cheby1' and self.isThereAnError == 0:
            self.COEFFS = signal.cheby1(order, rp, cutoff, filterType, output='sos')
        elif design == 'cheby2' and self.isThereAnError == 0:
            self.COEFFS = signal.cheby2(order, rs, cutoff, filterType, output='sos')

        return self.COEFFS

    def __init__(self, order, cutoff, filterType, design='butter', rp=1, rs=1, fs=0):
        self.COEFFS = self.createCoeffs(order, cutoff, filterType, design, rp, rs, fs)
        self.acc_input = np.zeros(len(self.COEFFS))
        self.acc_output = np.zeros(len(self.COEFFS))
        self.buffer1 = np.zeros(len(self.COEFFS))
        self.buffer2 = np.zeros(len(self.COEFFS))
        self.input = 0
        self.output = 0

    def filter(self, input):

        # len(COEFFS[0,:] == 1 means that there was an error in the generation
        # of the coefficients and the filtering should not be used
        if len(self.COEFFS[0, :]) > 1:

            self.input = input
            self.output = 0

            # The for loop creates a chain of second order filters according to
            # the order desired. If a 10th order filter is to be created the
            # loop will iterate 5 times to create a chain of 5 second order
            # filters.
            for i in range(len(self.COEFFS)):
                self.FIRCOEFFS = self.COEFFS[i][0:3]
                self.IIRCOEFFS = self.COEFFS[i][3:6]  #IIRCOEFFS[0] == 1 right?

                # Calculating the accumulated input consisting of the input and
                # the values coming from the feedbaack loops (delay buffers
                # weighed by the IIR coefficients).
                self.acc_input[i] = (self.input + self.buffer1[i] * -self.IIRCOEFFS[1] + self.buffer2[i] * -self.IIRCOEFFS[2])

                # Calculating the accumulated output provided by the accumulated
                # input and the values from the delay bufferes weighed by the
                # FIR coefficients.
                self.acc_output[i] = (self.acc_input[i] * self.FIRCOEFFS[0] + self.buffer1[i] * self.FIRCOEFFS[1] + self.buffer2[i] * self.FIRCOEFFS[2])

                # Shifting the values on the delay line: acc_input->buffer1->
                # buffer2
                self.buffer2[i] = self.buffer1[i]
                self.buffer1[i] = self.acc_input[i]

                self.input = self.acc_output[i]

            self.output = self.acc_output[i]

        return self.output



# ### Use Example: ###
# FilterMains = IIR2Filter(10, [55], 'lowpass', design='cheby1', rp=2, fs=1000)
# signal.cheby1(10, 2, [55/1000], 'lowpass', output='sos')
# signal.cheby1(10, 2, [55/1000], 'lowpass')
# mySignal = randn(1000)
# mySignalFiltered = randn(1000)
# for i in range(len(mySignal)):
#     mySignalFiltered[i] = FilterMains.filter(mySignal[i])
# plot(mySignal)
# plot(mySignalFiltered)






### Direct-Form I Try: ###
class IIR_layer_directformI(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100):
        super(IIR_layer_directformI, self).__init__()
        ### Coefficients: ###
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.FIR_coefficients = torch.zeros(number_of_FIR_coefficients)
        self.IIR_coefficients = torch.zeros(number_of_IIR_coefficients-1)
        ### Buffers: ###
        self.input_buffer = torch.zeros(max_number_of_coefficients) #[x(n), x(n-1), x(n-2)....]
        self.output_buffer = torch.zeros(max_number_of_coefficients) #[y(n), y(n-1), y(n-2)....]
        self.input = 0
        self.output = 0
        self.max_number_of_coefficients = max_number_of_coefficients;

    def filter(self, input):
        self.input_buffer = torch.zeros(input.shape[0], input.shape[1], max_number_of_coefficients)  # [x(n), x(n-1), x(n-2)....]
        self.output_buffer = torch.zeros(input.shape[0], input.shape[1], max_number_of_coefficients)  # [y(n), y(n-1), y(n-2)....]
        output = torch.zeros_like(input)
        B,H,C = input.shape

        ### Loop over pixel/row/col indices: ###
        for pixel_index in arange(H):
            ### Shift input and output buffers by 1 sample: ###
            for i in arange(self.max_number_of_coefficients):
                self.output_buffer[:,:,i] = self.output_buffer[:,:,i-1]  # time interpretable:  y(-n)=y(-n+1), y(-n+1)=y(-n+2), ..., y(-1)=y(0) ... remember the way the array are build are opposite to intuitive time passing sometimes
                self.input_buffer[:,:,i] = self.input_buffer[:,:,i-1]  # time interpretable: ... x(n-1)=x(n)   /  x(-n)=x(-n+1), x(-n+1)=x(-n+2), ..., x(-1)=x(0)

            ### Initialize first element in buffers (representing newest sample): ###
            self.input_buffer[:,:,0] = input[:,:,pixel_index]
            self.output_buffer[:,:,0] = 0

            ### FIR Part: ###
            for i in arange(len(self.FIR_coefficients)):
                self.output_buffer[:,:,0] += self.input_buffer[:,:,i]*self.FIR_coefficients[i]

            ### IIR Part: ###
            for i in arange(len(self.IIR_coefficients)):  #remember that by default usually IIR_coefficients[0]=1
                self.output_buffer[:,:,0] = self.output_buffer[:,:,0] - self.output_buffer[:,:,i]*self.IIR_coefficients[:,:,i]  #remember that in the paper it was said it makes sense that it's more stable
                                                                                                                                    # to instead have the filter in the cascaded version and have g=1-h as 0 and 1 order coefficients

            ### Assign final result to output signal: ###
            output[:,:, pixel_index] = self.output_buffer[:,:,0]






### Direct-Form II Try: ###
class IIR_layer_directfromII(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100):
        super(IIR_layer_directfromII, self).__init__()
        ### Coefficients: ###
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.FIR_coefficients = torch.zeros(number_of_FIR_coefficients)
        self.IIR_coefficients = torch.zeros(number_of_IIR_coefficients)
        ### Buffers: ###
        self.input_buffer = 0
        self.output_buffer = 0
        self.w_buffer = torch.zeros(max_number_of_coefficients)
        self.input = 0
        self.output = 0
        self.max_number_of_coefficients = max_number_of_coefficients;

    def filter(self, input):
        self.input_buffer = torch.zeros(input.shape[0], input.shape[1], 1)
        self.output_buffer = torch.zeros(input.shape[0], input.shape[1], 1)
        self.w_buffer = torch.zeros(input.shape[0], input.shape[1], max_number_of_coefficients)
        output = torch.zeros_like(input)
        B, H, C = input.shape

        ### Loop over pixel/row/col indices: ###
        for pixel_index in arange(H):
            ### Shift input and output buffers by 1 sample: ###
            for i in arange(self.max_number_of_coefficients):
                self.w_buffer[:, :, i] = self.w_buffer[:, :, i - 1]  # time interpretable:  y(-n)=y(-n+1), y(-n+1)=y(-n+2), ..., y(-1)=y(0) ... remember the way the array are build are opposite to intuitive time passing sometimes

            ### Initialize first element in buffers (representing newest sample): ###
            self.input_buffer[:, :, 0] = input[:, :, pixel_index]
            self.output_buffer[:, :, 0] = 0

            ### IIR Part: ###
            for i in arange(len(self.IIR_coefficients)):
                self.w_buffer[:, :, 0] += self.w_buffer[:, :, i] * self.IIR_coefficients[i] # remember that by default usually IIR_coefficients[0]=1

            ### FIR Part: ###
            for i in arange(len(self.FIR_coefficients)):
                self.output_buffer[:, :, 0] = self.output_buffer[:, :, 0] - self.w_buffer[:, :, i] * self.IIR_coefficients[i]  # remember that in the paper it was said it makes sense that it's more stable   # to instead have the filter in the cascaded version and have g=1-h as 0 and 1 order coefficients

            ### Assign final result to output signal: ###
            output[:, :, pixel_index] = self.output_buffer[:, :, 0]






### Direct-Form II Transposed Try: ###
class IIR_layer_directfromIItransposed(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100):
        super(IIR_layer_directfromIItransposed, self).__init__()
        ### Coefficients: ###
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.FIR_coefficients = torch.zeros(number_of_FIR_coefficients)
        self.IIR_coefficients = torch.zeros(number_of_IIR_coefficients)
        ### Buffers: ###
        self.input_buffer = 0
        self.output_buffer = 0
        self.w_buffer = torch.zeros(max_number_of_coefficients)
        self.input = 0
        self.output = 0
        self.max_number_of_coefficients = max_number_of_coefficients;

    def filter(self, input):
        self.input_buffer = torch.zeros(input.shape[0], input.shape[1], 1)
        self.output_buffer = torch.zeros(input.shape[0], input.shape[1], 1)
        self.z_buffer = torch.zeros(input.shape[0], input.shape[1], max_number_of_coefficients)
        output = torch.zeros_like(input)
        B, H, C = input.shape

        ### Loop over pixel/row/col indices: ###
        for pixel_index in arange(H):
            output[:,:,pixel_index] = self.FIR_coefficients[0]*input[:,:,0] + self.z_buffer[0]
            for i in arange(self.max_number_of_coefficients-1):  #it seems self.z_buffer[-1] never updates this way!!!!...is there no shifting or something?
                self.z_buffer[i] = self.z_buffer[i+1] + self.FIR_coefficients[i+1]*input[:,:,pixel_index] - self.IIR_coefficients[i+1]*output[:,:,pixel_index]






###############################################################################################################################################################################################################################################################
### Cascaded 1st-order filters Form Try: ###
class IIR_layer_cascaded(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True):
        super(IIR_layer_cascaded, self).__init__()
        ### Coefficients: ###
        #TODO: implement possiblity of outside IIR_coefficients map like in the paper!
        #TODO: implement possibility of horizontal,vertical & reverse directions
        #TODO: implement Conv2d_IIR layer which outputs feature maps....is there a parallel able to be done like Conv2d incorporates all input features for each output map?....
        #      i think yes -> have a different IIR filter for each input and combine for final single output...
        #      maybe i can even simply to turn on the same iir on each input map and have the number of IIR filter as the number of outputs thus having weights of [output_channels,1,3,1] and [output_channels,1,1,3]
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.H = H
        self.flag_stable_filter_learning = flag_stable_filter_learning
        self.flag_bound_p_below_unit_circle = flag_bound_p_below_unit_circle
        self.FIR_coefficients = nn.ParameterList()
        self.IIR_numerator_coefficients = nn.ParameterList()
        self.IIR_denominator_coefficients = nn.ParameterList()

        default_IIR_initial_value = 0.1
        for i in arange(max_number_of_coefficients):
            self.FIR_coefficients.append(nn.Parameter(torch.Tensor([default_IIR_initial_value])))  #TODO: choose an initialization...maybe even use some default designed known IIR filter like a certain lowpass/bandpass/highpass whatever....maybe use something that passes almost everything
        for i in arange(max_number_of_coefficients):
            self.IIR_numerator_coefficients.append(nn.Parameter(torch.Tensor([default_IIR_initial_value])))
            self.IIR_denominator_coefficients.append(nn.Parameter(torch.Tensor([default_IIR_initial_value])))
        ### Set learnable and unlearnable/frozen parameters according to wanted FIR and IIR orders (for instance if i want pure IIR filter and FIR order / number_of_coefficients = 0, i should set them so...or if FIR or IIR orders aren't the same): ###
        for i in arange(max_number_of_coefficients-number_of_FIR_coefficients):
            self.FIR_coefficients[-(i+1)] = nn.Parameter(torch.Tensor([0]))
            self.FIR_coefficients[-(i + 1)].requires_grad = False
        for i in arange(max_number_of_coefficients-number_of_IIR_coefficients):
            self.FIR_coefficients[-(i+1)] = nn.Parameter(torch.Tensor([0]))
            self.FIR_coefficients[-(i + 1)].requires_grad = False
        ### Buffers: ###
        self.input_buffer = 0
        self.output_buffer = 0
        self.w_buffer = torch.zeros(max_number_of_coefficients)
        self.input = 0
        self.output = 0
        self.max_number_of_coefficients = max_number_of_coefficients;


    def forward(self, input):
        input_buffer = torch.zeros(input.shape[0], input.shape[1], self.H)
        output_buffer = torch.zeros(input.shape[0], input.shape[1], self.H)
        input_buffer.copy_(input);
        # output = torch.zeros_like(input)
        B, C, H = input.shape

        if self.flag_bound_p_below_unit_circle:
            for i in arange(len(self.IIR_denominator_coefficients)):
                self.IIR_denominator_coefficients[i].clamp(-0.99,0.99)

        ### Loop cascades: ###
        for cascade_index in arange(self.max_number_of_coefficients):   #max_number_of_coefficients = number of cascades
            ### ReInitialize output buffer for current cascade: ###
            output_buffer = torch.zeros(input.shape[0], input.shape[1], H)

            ### Loop over pixel/row/col indices: ###
            output_buffer[:, :, 0] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, 0] + self.IIR_denominator_coefficients[cascade_index] * output_buffer[:, :, 0]  #basically having "repeat" boundary conditions...don't think this is critical
            for pixel_index in arange(1,H):
                if self.flag_stable_filter_learning:
                    output_buffer[:, :, pixel_index] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, pixel_index] +\
                                                            (1-self.IIR_numerator_coefficients[cascade_index]) * output_buffer[:, :, pixel_index - 1] + \
                                                            self.FIR_coefficients[cascade_index] * input_buffer[:, :, pixel_index - 1]
                else:
                    output_buffer[:,:,pixel_index] = self.IIR_numerator_coefficients[cascade_index]*input_buffer[:,:,pixel_index] + \
                                                          self.IIR_denominator_coefficients[cascade_index]*output_buffer[:,:,pixel_index-1] + \
                                                          self.FIR_coefficients[cascade_index]*input_buffer[:,:,pixel_index-1]

            ### Assign input for next cascade to be current cascade output: ###
            input_buffer = output_buffer

        return output_buffer








### Cascaded 1st-order filters Form Try: ###
class IIR_layer_cascaded_1D(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100, flag_reverse_direction=False, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True):
        super(IIR_layer_cascaded_1D, self).__init__()
        ### Coefficients: ###
        #TODO: implement possiblity of outside IIR_coefficients map like in the paper!
        #TODO: implement possibility of horizontal,vertical & reverse directions
        #TODO: implement Conv2d_IIR layer which outputs feature maps....is there a parallel able to be done like Conv2d incorporates all input features for each output map?....
        #      i think yes -> have a different IIR filter for each input and combine for final single output...
        #      maybe i can even simply to turn on the same iir on each input map and have the number of IIR filter as the number of outputs thus having weights of [output_channels,1,3,1] and [output_channels,1,1,3]
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.H = H
        self.flag_stable_filter_learning = flag_stable_filter_learning
        self.flag_bound_p_below_unit_circle = flag_bound_p_below_unit_circle
        self.FIR_coefficients = nn.ParameterList()
        self.IIR_numerator_coefficients = nn.ParameterList()
        self.IIR_denominator_coefficients = nn.ParameterList()

        ### Initialize FIR & IIR coefficients: ###
        default_IIR_initial_value = 0.1
        for i in arange(max_number_of_coefficients):
            self.FIR_coefficients.append(nn.Parameter(torch.Tensor([default_IIR_initial_value])))  #TODO: choose an initialization...maybe even use some default designed known IIR filter like a certain lowpass/bandpass/highpass whatever....maybe use something that passes almost everything
        for i in arange(max_number_of_coefficients):
            self.IIR_numerator_coefficients.append(nn.Parameter(torch.Tensor([default_IIR_initial_value])))
            self.IIR_denominator_coefficients.append(nn.Parameter(torch.Tensor([default_IIR_initial_value])))
        ### Set learnable and unlearnable/frozen parameters according to wanted FIR and IIR orders (for instance if i want pure IIR filter and FIR order / number_of_coefficients = 0, i should set them so...or if FIR or IIR orders aren't the same): ###
        for i in arange(max_number_of_coefficients-number_of_FIR_coefficients):
            self.FIR_coefficients[-(i+1)] = nn.Parameter(torch.Tensor([0]))
            self.FIR_coefficients[-(i + 1)].requires_grad = False
        for i in arange(max_number_of_coefficients-number_of_IIR_coefficients):
            self.FIR_coefficients[-(i+1)] = nn.Parameter(torch.Tensor([0]))
            self.FIR_coefficients[-(i + 1)].requires_grad = False

        ### Buffers: ###
        self.input_buffer = 0
        self.output_buffer = 0
        self.w_buffer = torch.zeros(max_number_of_coefficients)
        self.input = 0
        self.output = 0
        self.max_number_of_coefficients = max_number_of_coefficients;
        self.flag_reverse_direction = flag_reverse_direction
        if self.flag_reverse_direction:
            #"Reverse direction":
            self.initial_index = H-1
            self.stop_index = 0
            self.previous_time_step_adder = -1
            self.pixel_range_array = list(reversed(arange(H-1)))
        else:
            #"Regular":
            self.initial_index = 0
            self.stop_index = H
            self.previous_time_step_adder = 1
            self.pixel_range_array = list(arange(1, H))

    def forward(self, input):
        input_buffer = torch.zeros(input.shape[0], input.shape[1], self.H)
        output_buffer = torch.zeros(input.shape[0], input.shape[1], self.H)
        input_buffer.copy_(input);
        # output = torch.zeros_like(input)
        B, C, H = input.shape

        if self.flag_bound_p_below_unit_circle:
            for i in arange(len(self.IIR_denominator_coefficients)):
                self.IIR_denominator_coefficients[i].clamp(-0.99,0.99)

        ### Loop cascades: ###
        for cascade_index in arange(self.max_number_of_coefficients):   #max_number_of_coefficients = number of cascades
            ### ReInitialize output buffer for current cascade: ###
            output_buffer = torch.zeros(input.shape[0], input.shape[1], H)

            ### Loop over pixel/row/col indices: ###
            output_buffer[:, :, self.initial_index] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, self.initial_index] + self.IIR_denominator_coefficients[cascade_index] * output_buffer[:, :, self.initial_index]  #basically having "repeat" boundary conditions...don't think this is critical
            for pixel_index in self.pixel_range_array:
                if self.flag_stable_filter_learning:
                    output_buffer[:, :, pixel_index] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, pixel_index] +\
                                                            (1-self.IIR_numerator_coefficients[cascade_index]) * output_buffer[:, :, pixel_index - self.previous_time_step_adder] + \
                                                            self.FIR_coefficients[cascade_index] * input_buffer[:, :, pixel_index - self.previous_time_step_adder]
                else:
                    output_buffer[:,:,pixel_index] = self.IIR_numerator_coefficients[cascade_index]*input_buffer[:,:,pixel_index] + \
                                                          self.IIR_denominator_coefficients[cascade_index]*output_buffer[:,:,pixel_index - self.previous_time_step_adder] + \
                                                          self.FIR_coefficients[cascade_index]*input_buffer[:,:,pixel_index - self.previous_time_step_adder]

            ### Assign input for next cascade to be current cascade output: ###
            input_buffer = output_buffer

        return output_buffer


# ### Example of use: ###
# layer_straight = IIR_layer_cascaded_1D(number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100, flag_reverse_direction=False, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True)
# layer_reversed = IIR_layer_cascaded_1D(number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100, flag_reverse_direction=True, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True)
# input_tensor_straight = torch.Tensor(randn(1,1,100))
# input_tensor_reversed = input_tensor_straight.flip([2])
#
# output_tensor_straight_input_straight_layer = layer_straight(input_tensor_straight)
# output_tensor_straight_input_reversed_layer = layer_reversed(input_tensor_straight)
# output_tensor_reversed_input_straight_layer = layer_straight(input_tensor_reversed)
# output_tensor_reversed_input_reversed_layer = layer_reversed(input_tensor_reversed)
# figure(1)
# plot_torch(output_tensor_straight_input_straight_layer)
# title('straight input straight layer')
# figure(2)
# plot_torch(output_tensor_straight_input_reversed_layer)
# title('straight input reversed layer')
# figure(3)
# plot_torch(output_tensor_reversed_input_straight_layer)
# title('reversed input straight layer')
# figure(4)
# plot_torch(output_tensor_reversed_input_reversed_layer)
# title('reversed input reversed layer')











### Cascaded 1st-order filters Form Try: ###
class IIR_layer_cascaded_2D(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, flag_horizontal_vertical=0, flag_reverse_direction=False, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True):
        super(IIR_layer_cascaded_2D, self).__init__()
        ### I assume


        ### Coefficients: ###
        #TODO: implement possiblity of outside IIR_coefficients map like in the paper!
        #TODO: implement possibility of horizontal,vertical & reverse directions
        #TODO: implement Conv2d_IIR layer which outputs feature maps....is there a parallel able to be done like Conv2d incorporates all input features for each output map?....
        #      i think yes -> have a different IIR filter for each input and combine for final single output...
        #      maybe i can even simply to turn on the same iir on each input map and have the number of IIR filter as the number of outputs thus having weights of [output_channels,1,3,1] and [output_channels,1,1,3]
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.flag_horizontal_vertical = flag_horizontal_vertical
        self.flag_stable_filter_learning = flag_stable_filter_learning
        self.flag_bound_p_below_unit_circle = flag_bound_p_below_unit_circle
        self.FIR_coefficients = nn.ParameterList()
        self.IIR_numerator_coefficients = nn.ParameterList()
        self.IIR_denominator_coefficients = nn.ParameterList()

        ### Initialize FIR & IIR coefficients: ###
        default_IIR_initial_value = 0.2
        epsilon = 0.05
        for i in arange(max_number_of_coefficients):
            random_number = np.random.uniform(0+epsilon, 1-epsilon)
            self.FIR_coefficients.append(nn.Parameter(torch.Tensor([random_number])))  #TODO: choose an initialization...maybe even use some default designed known IIR filter like a certain lowpass/bandpass/highpass whatever....maybe use something that passes almost everything
        for i in arange(0, max_number_of_coefficients):
            random_number = np.random.uniform(0 + epsilon, 1 - epsilon)
            self.IIR_numerator_coefficients.append(nn.Parameter(torch.Tensor([random_number])))
            random_number = np.random.uniform(0 + epsilon, 1 - epsilon)
            self.IIR_denominator_coefficients.append(nn.Parameter(torch.Tensor([random_number])))
        ### Set learnable and unlearnable/frozen parameters according to wanted FIR and IIR orders (for instance if i want pure IIR filter and FIR order / number_of_coefficients = 0, i should set them so...or if FIR or IIR orders aren't the same): ###
        for i in arange(max_number_of_coefficients-number_of_FIR_coefficients):
            self.FIR_coefficients[-(i+1)] = nn.Parameter(torch.Tensor([0]))
            self.FIR_coefficients[-(i + 1)].requires_grad = False
        for i in arange(max_number_of_coefficients-number_of_IIR_coefficients):
            self.FIR_coefficients[-(i+1)] = nn.Parameter(torch.Tensor([0]))
            self.FIR_coefficients[-(i + 1)].requires_grad = False

        ### Buffers: ###
        self.input_buffer = 0
        self.output_buffer = 0
        self.w_buffer = torch.zeros(max_number_of_coefficients)
        self.input = 0
        self.output = 0
        self.max_number_of_coefficients = max_number_of_coefficients;
        self.flag_reverse_direction = flag_reverse_direction
        self.flag_first = True



    def forward(self, input):
        # output = torch.zeros_like(input)
        B, C, H, W = input.shape
        #TODO: i don't think i have to allocate input_buffer, i should be able to continously change input array each cascade
        input_buffer = torch.zeros_like(input)
        output_buffer = torch.zeros_like(input)
        input_buffer.copy_(input);

        ### Set parameters if this is the first forward: ###
        if self.flag_first:
            self.flag_first = False
            if self.flag_reverse_direction:
                #"Reverse direction":
                #(1). Rows:
                self.initial_index_rows = H-1
                self.stop_index_rows = 0
                self.previous_time_step_adder = -1
                self.pixel_range_array_rows = list(reversed(arange(H-1)))
                # (1). Cols:
                self.initial_index_cols = W - 1
                self.stop_index_cols = 0
                self.previous_time_step_adder = -1
                self.pixel_range_array_cols = list(reversed(arange(W - 1)))
            else:
                #"Regular":
                #(1). Rows:
                self.initial_index_rows = 0
                self.stop_index_rows = H
                self.previous_time_step_adder = 1
                self.pixel_range_array_rows = list(arange(1, H))
                # (1). Cols:
                self.initial_index_cols = 0
                self.stop_index_cols = W
                self.previous_time_step_adder = 1
                self.pixel_range_array_cols = list(arange(1, W))


        if self.flag_bound_p_below_unit_circle:
            for i in arange(len(self.IIR_denominator_coefficients)):
                self.IIR_denominator_coefficients[i].clamp(-0.99,0.99)

        ### Loop cascades: ###
        for cascade_index in arange(self.max_number_of_coefficients):   #max_number_of_coefficients = number of cascades
            ### ReInitialize output buffer for current cascade: ###
            output_buffer = torch.zeros_like(input_buffer)

            #####################################################
            if self.flag_horizontal_vertical==0: # horizontal==left to right / right to left
                ### Initial boundary conditions: ###
                # output_buffer[:, :, :, self.initial_index_cols] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, :, self.initial_index_cols] + \
                #                                              self.IIR_denominator_coefficients[cascade_index] * output_buffer[:, :, :, self.initial_index_cols]  #basically having "repeat" boundary conditions...don't think this is critical
                output_buffer[:, :, :, self.initial_index_cols] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, :, self.initial_index_cols] + \
                                                                  (1-self.IIR_numerator_coefficients[cascade_index]) * output_buffer[:, :, :, self.initial_index_cols]  # basically having "repeat" boundary conditions...don't think this is critical
                for pixel_index in self.pixel_range_array_cols:
                    if self.flag_stable_filter_learning:
                        output_buffer[:, :, :, pixel_index] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, :, pixel_index] +\
                                                                (1-self.IIR_numerator_coefficients[cascade_index]) * output_buffer[:, :, :, pixel_index - self.previous_time_step_adder] + \
                                                                self.FIR_coefficients[cascade_index] * input_buffer[:, :, :, pixel_index - self.previous_time_step_adder]
                    else:
                        output_buffer[:,:,:, pixel_index] = self.IIR_numerator_coefficients[cascade_index]*input_buffer[:, :, :, pixel_index] + \
                                                              self.IIR_denominator_coefficients[cascade_index]*output_buffer[:, :, :, pixel_index - self.previous_time_step_adder] + \
                                                              self.FIR_coefficients[cascade_index]*input_buffer[:, :, :, pixel_index - self.previous_time_step_adder]
            #####################################################
            else: #vertical==up to bottom / bottom to up
                ### Initial boundary conditions: ###
                # output_buffer[:, :, self.initial_index_rows, :] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, self.initial_index_rows, :] + \
                #                                                   self.IIR_denominator_coefficients[cascade_index] * output_buffer[:, :, self.initial_index_rows, :]  # basically having "repeat" boundary conditions...don't think this is critical
                output_buffer[:, :, self.initial_index_rows, :] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, self.initial_index_rows, :] +\
                                                                  (1-self.IIR_numerator_coefficients[cascade_index]) * output_buffer[:, :, self.initial_index_rows, :]  # basically having "repeat" boundary conditions...don't think this is critical
                for pixel_index in self.pixel_range_array_rows:
                    if self.flag_stable_filter_learning:
                        output_buffer[:, :, pixel_index, :] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, pixel_index, :] + \
                                                              (1 - self.IIR_numerator_coefficients[cascade_index]) * output_buffer[:, :, pixel_index - self.previous_time_step_adder, :] +\
                                                              self.FIR_coefficients[cascade_index] * input_buffer[:, :, pixel_index - self.previous_time_step_adder, :]
                    else:
                        output_buffer[:, :, pixel_index, :] = self.IIR_numerator_coefficients[cascade_index] * input_buffer[:, :, pixel_index, :] + \
                                                              self.IIR_denominator_coefficients[cascade_index] * output_buffer[:, :, pixel_index - self.previous_time_step_adder, :] + \
                                                              self.FIR_coefficients[cascade_index] * input_buffer[:, :, pixel_index - self.previous_time_step_adder, :]
            #####################################################

            ### Assign input for next cascade to be current cascade output: ###
            input_buffer = output_buffer

        return output_buffer


# layer = IIR_layer_cascaded_2D(number_of_FIR_coefficients=0, number_of_IIR_coefficients=4, flag_horizontal_vertical=1, flag_reverse_direction=False, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True)
# layer2 = IIR_layer_cascaded_2D(number_of_FIR_coefficients=0, number_of_IIR_coefficients=4, flag_horizontal_vertical=1, flag_reverse_direction=True, flag_stable_filter_learning=True, flag_bound_p_below_unit_circle=True)
# input_tensor = torch.Tensor(randn(1,3,100,100))
# input_tensor_padded = nn.ReflectionPad2d(14)(input_tensor)
# output_tensor = (layer(input_tensor_padded))
# output_tensor2 = (layer(output_tensor.flip(2))).flip(2)
# output_tensor2 = Crop_Center_Layer(100,100)(output_tensor2)
# # output_tensor3 = (layer2(input_tensor))
# # output_tensor2 = (layer2(output_tensor)) #matlab's filtfilt....probably doesn't fit HardWare implementation because it's line-by-line and i don't have access to entire image....as far as SOC goes they said i can have order in W as much as i want but H order i think lower than 4
#
# ### Images: ###
# figure(1)
# imshow_torch(output_tensor[0,:,:,:])
# # imshow_torch(output_tensor.flip(2)[0,:,:,:])
# figure(2)
# imshow_torch(output_tensor2[0,:,:,:])
#
# ### Plots: ###
# figure(1)
# plot_torch(input_tensor[0,0,:,0])
# figure(2)
# plot_torch(output_tensor[0,0,:,0])
# # figure(3)
# # plot_torch(output_tensor3[0,0,:,0])
# figure(4)
# plot_torch(output_tensor2[0,0,:,0])
# ###############################################################################################################################################################################################################################################################












### Incorrect Version i tried to base off of the above python/numpy cascaded sos version: ###
class IIR_layer(nn.Module):
    ### For now only single direction for 1D signals: ###
    def __init__(self, number_of_FIR_coefficients=3, number_of_IIR_coefficients=3, H=100):
        super(IIR_layer, self).__init__()
        ### Coefficients: ###
        max_number_of_coefficients = max(number_of_FIR_coefficients, number_of_IIR_coefficients)
        self.FIR_coefficients = torch.zeros(number_of_FIR_coefficients)
        self.IIR_coefficients = torch.zeros(number_of_IIR_coefficients-1)
        ### Buffers: ###
        self.accumulated_input = torch.zeros(max_number_of_coefficients)
        self.accumulated_output = torch.zeros(max_number_of_coefficients)
        self.input_buffer = torch.zeros(max_number_of_coefficients)
        self.input_buffer = torch.zeros(max_number_of_coefficients)
        self.input = 0
        self.output = 0


    def filter(self, input):
        self.input = input
        self.output = 0
        B,H,C = input.shape

        for pixel_index in arange(H):
            # Calculating the accumulated input consisting of the input and the values coming from the feedbaack loops (delay buffers weighed by the IIR coefficients).
            self.accumulated_input = self.input[pixel_index] ### by definition IIR_coefficients[0]==1!!!
            for IIR_index in arange(len(self.IIR_coefficients)):
                self.accumulated_input += self.input_buffer[IIR_index] * -self.IIR_coefficients[IIR_index]

            # Calculating the accumulated output provided by the accumulated input and the values from the delay bufferes weighed by the FIR coefficients.
            self.accumulated_output = self.accumulated_input * self.FIR_coefficients[0]
            for FIR_index in arange(len(self.FIR_coefficients)-1):
                self.accumulated_output[pixel_index] += self.input_buffer[FIR_index] * self.FIR_coefficients[FIR_index+1]

            # Shifting the values on the delay line:
            #(1). Shift buffer by 1 and insert accumulated input to the last index:
            self.input_buffer[0:max_number_of_coefficients-1] = self.input_buffer[1:max_number_of_coefficients]
            self.input_buffer[-1] = self.accumulated_input[pixel_index]
            #(2). Replace input by accumulated output:
            self.input[pixel_index] = self.accumulated_output[pixel_index]










