from utils import push_data_to_queue, get_data_from_queue


class KayaManager:
    def __init__(self, bgs_input_queue, bgs_input_cond, ransac_input_queue, ransac_input_cond,
                 output_queue, output_cond, main_msg_queue, main_msg_cond):
        # Input objects for BGS
        self.bgs_input_queue = bgs_input_queue
        self.bgs_input_cond = bgs_input_cond

        # Input objects for ransac
        self.ransac_input_queue = ransac_input_queue
        self.ransac_input_cond = ransac_input_cond

        # Output objects
        self.output_queue = output_queue
        self.output_cond = output_cond

        # Main process communication objects
        self.main_msg_queue = main_msg_queue
        self.main_msg_cond = main_msg_cond

    def push_data_to_bgs(self, data):
        """
        PRODUCER - Push next batch of frames for BGS
        """
        push_data_to_queue(data, self.bgs_input_queue, self.bgs_input_cond)

    def get_data_for_bgs(self):
        """
        CONSUMER - Get data to perform BGS on
        """
        result = get_data_from_queue(self.bgs_input_queue, self.bgs_input_cond)
        return result

    def push_data_to_ransac(self, data):
        """
        PRODUCER - Push next batch of frames for Ransac
        """
        push_data_to_queue(data, self.ransac_input_queue, self.ransac_input_cond)

    def get_data_for_ransac(self):
        """
        CONSUMER - Get data to perform Ransac on
        """
        result = get_data_from_queue(self.ransac_input_queue, self.ransac_input_cond)
        return result

    def push_output_data(self, data):
        """
        PRODUCER - Push data to final output queue
        """
        push_data_to_queue(data, self.output_queue, self.output_cond)

    def get_output_data(self):
        """
        CONSUMER - Get data from final output queue
        """
        result = get_data_from_queue(self.output_queue, self.output_cond)
        return result

    def publish_message_to_main(self, message):
        """
        Put a message in the main message communication queue and notify
        """
        push_data_to_queue(message, self.main_msg_queue, self.main_msg_cond)

    def get_main_message(self):
        """
        Block and wait (non-busy) for new message on main communication queue
        """
        result = get_data_from_queue(self.main_msg_queue, self.main_msg_cond)
        return result
