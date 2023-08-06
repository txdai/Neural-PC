img_input = TF.to_tensor(self.crop(img_input))
img_target = TF.to_tensor(self.crop(img_target))
structuring_element = torch.ones((1, 1, 41, 41))
img_input = img_input.unsqueeze_(0)
img_dilated = torch.nn.functional.conv2d(img_input, structuring_element, padding=20) > 0
img_target = img_target * img_dilated.squeeze_(0).float()