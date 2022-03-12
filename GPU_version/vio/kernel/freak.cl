void meanIntensity( image2d_t _image, image2d_t _integral, float kp_x, float kp_y,
							  int scale, int rot, int* point, float3* patternLookupPtr, uchar* output)
{
    // Prepare a suitable OpenCL image sampler.
    sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    // get point position in image
    float3* FreakPoint = patternLookupPtr + (scale * FREAK_NB_ORIENTATION * FREAK_NB_POINTS + rot * FREAK_NB_POINTS + (*point));
    float xf = FreakPoint->x + kp_x;
    float yf = FreakPoint->y + kp_y;
    int x = (int)xf;
    int y = (int)yf;

    // get the sigma:
    float radius = FreakPoint->z;//sigma

    // calculate output:
    if( radius < 0.5 )
    {
        // interpolation multipliers:
        const int r_x = (int)((xf - x) * 1024);
        const int r_y = (int)((yf - y) * 1024);
        const int r_x_1 = (1024 - r_x);
        const int r_y_1 = (1024 - r_y);
        uint ret_val;
        // linear interpolation:
        ret_val = (r_x_1 * r_y_1 * (int)(read_imageui(_image, sampler, (int2)(y  , x  ))).x)
                + (r_x   * r_y_1 * (int)(read_imageui(_image, sampler, (int2)(y  , x+1))).x)
                + (r_x_1 * r_y   * (int)(read_imageui(_image, sampler, (int2)(y+1, x  ))).x)
                + (r_x   * r_y   * (int)(read_imageui(_image, sampler, (int2)(y+1, x+1))).x);
        // return the rounded mean
        ret_val += 2 * 1024 * 1024;
        output[*point] = (uchar)(ret_val / (4 * 1024 * 1024));
    }

    // calculate borders
    const int x_left   = round(xf - radius);
    const int y_top    = round(yf - radius);
    const int x_right  = round(xf + radius + 1); // integral image is 1px wider
    const int y_bottom = round(yf + radius + 1); // integral image is 1px higher
    int ret_val;

    ret_val  = read_imageui(_integral, sampler, (int2)(y_bottom, x_right)).x; // bottom right corner
    ret_val -= read_imageui(_integral, sampler, (int2)(y_bottom, x_left)).x; // bottom left corner
    ret_val += read_imageui(_integral, sampler, (int2)(y_top,    x_left)).x; // top left corner
    ret_val -= read_imageui(_integral, sampler, (int2)(y_top,    x_right)).x; // top right corner
    const int area = (x_right - x_left) * (y_bottom - y_top);
    ret_val = (ret_val + area / 2) / area;
    output[*point] = (uchar)(ret_val);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void freak(
    __global		image2d_t*	image, // grayescale image
	__global		image2d_t*	imgIntegral,
    __global		float4*			keypoints, // ptx, pty, angle
    __global		uchar*			descriptors, // kp_size * (64 = NB_PAIRS/8) - CV_8U with zero values
	__global		int*				patternSizes, // array size : NB_SCALES = 64
	__global		int4*				orientationPairs, // array size : FREAK_NB_ORIENTATION = 256
	__global		float3*			patternLookup, // array size : NB_SCALES(64) * FREAK_NB_ORIENTATION(256) * FREAK_NB_POINTS(43)
	__global		uchar2*			descriptionPairs, // array size : NB_PAIRS = 512
	__global		int*				nOctaves,
	__global		int*				kpScaleIdx //array size = kp_size - used to save pattern scale index corresponding to each keypoints
	)
{
    const float sizeCst = (float)(NB_SCALES / (FREAK_LOG2 * *nOctaves));
    uchar pointsValue[FREAK_NB_POINTS];
    int thetaIdx = 0;
    int direction0;
    int direction1;

	int k = get_global_id(0);

    kpScaleIdx[k] = max( (int)(log(7.0 / FREAK_SMALLEST_KP_SIZE) * sizeCst + 0.5) ,0);
    if( kpScaleIdx[k] >= NB_SCALES )
        kpScaleIdx[k] = NB_SCALES - 1;

    if( keypoints[k].x <= patternSizes[kpScaleIdx[k]] || // check if the description at this specific position and scale fits inside the image
        keypoints[k].y <= patternSizes[kpScaleIdx[k]] ||
        keypoints[k].x >= get_image_dim(image[(int)keypoints[k].w]).y - patternSizes[kpScaleIdx[k]] ||
        keypoints[k].y >= get_image_dim(image[(int)keypoints[k].w]).x - patternSizes[kpScaleIdx[k]])
		return;

        // get the points intensity value in the un-rotated pattern
        for( int i = FREAK_NB_POINTS; i--; )
        {
            meanIntensity(image[(int)keypoints[k].w], imgIntegral[(int)keypoints[k].w], keypoints[k].x, keypoints[k].y,
                          *(kpScaleIdx + k), 0, &i, patternLookup, pointsValue);
        }
        direction0 = 0;
        direction1 = 0;
        for( int m = 45; m--; )
        {
            // iterate through the orientation pairs
            const int delta = (pointsValue[ orientationPairs[m].w ] - pointsValue[ orientationPairs[m].x ]);
            direction0 += delta * (orientationPairs[m].y) / 2048; // weight_dx
            direction1 += delta * (orientationPairs[m].z) / 2048; // weight_dy
        }

        thetaIdx = round(FREAK_NB_ORIENTATION *
						(float)(atan2((float)direction1,(float)direction0) * (180.0 / 3.141592653589793238462643383279502884197169399375)) *
						(1 / 360.0));

        if( thetaIdx < 0 )
            thetaIdx += FREAK_NB_ORIENTATION;

        if( thetaIdx >= FREAK_NB_ORIENTATION )
            thetaIdx -= FREAK_NB_ORIENTATION;

        // extract descriptor at the computed orientation
        for( int i = FREAK_NB_POINTS; i--; )
        {
            meanIntensity(image[(int)keypoints[k].w], imgIntegral[(int)keypoints[k].w], keypoints[k].x, keypoints[k].y,
                          *(kpScaleIdx + k), thetaIdx, &i, patternLookup, pointsValue);
        }

        // Extract descriptor
		int cnt = 0;
        for( int n = 0; n<4; n++ )
        {
			for (int m = 8; m >= 1; m--)
			{
				for(int kk = (n*16)+15; kk >= (n*16); --kk, ++cnt)
				{
					descriptors[(int)keypoints[k].w*300*64 + (k * 64) + kk] *= 2;
					if (pointsValue[(int)descriptionPairs[cnt].x] >= pointsValue[(int)descriptionPairs[cnt].y])
						descriptors[(int)keypoints[k].w*300*64 + (k * 64) + kk] += 1;
				}
			}
        }
}