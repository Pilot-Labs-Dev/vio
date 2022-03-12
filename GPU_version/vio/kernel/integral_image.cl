__kernel void integral_image(
    __read_only	    image2d_t*	int_img, // grayescale image - height X width
	__write_only	image2d_t*	out_img // (height+1) X (width+1) with zero values
	)
{
	sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	int k = get_global_id(0)
    int2 coord = (int2)(get_global_id(1)+1, get_global_id(2)+1);
	int res = 0;
	for (int i = 0; i < coord.x; i++)
		for (int j = 0; j < coord.y; j++)
			res += read_imageui(int_img[k], sampler, (int2)(i, j)); // bottom right corner
	write_imageui(out_img[k], coord, res);
}