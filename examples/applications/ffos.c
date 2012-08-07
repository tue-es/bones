//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example (real) application: 'Fast Focus
// on Structures' (FFOS). For more information on the application or on Bones
// please use the contact information below.
//
// == More information on the FFOS application
// Contact............Yifan He / Zhenyu Ye
// Web address........http://zhenyu-ye.net/publications/acivs2011/yifan2011acivs.pdf
// 
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/ffos.c
// Author.............Cedric Nugteren
// Last modified on...22-May-2012
//

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//########################################################################
//### Defines
//########################################################################

#define XVECTORS 10
#define YVECTORS 10

//########################################################################
//### Forward declarations
//########################################################################

void SaveBMPFile(unsigned char ** image, const char * outputdestination, int width, int height);
unsigned char ** LoadBMPFile(int *width, int *height);
void CPU_FindCenters(int* vector, int *coordinates, int size);
void CPU_Visualize(unsigned char** image0, int* Xcoordinates, int* Ycoordinates, unsigned char **image3, int width, int height);
void CPU_BCV(int *histogram, float *BCVtable, int size);

//########################################################################
//### Global variables
//########################################################################

int messages = 2;

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {

	// Declare loop variables
	int i,h,w,a;

	// Set other variables
	int threshold = 0;
	int hist[256];
	for (i=0;i<256;i++) {	hist[i] = 0; }	
	float * BCVtable = (float *)malloc(256*sizeof(float));

	// Loading image0 from disk
	if (messages == 2) { printf("### Loading image0 from disk.\n"); }
	int width = 0;
	int height = 0;
	unsigned char ** image0 = LoadBMPFile(&width, &height);
	
	// Create space for image1
	if (messages == 2) { printf("### Allocating space for image1.\n"); }
	unsigned char ** image1 = (unsigned char **)malloc(width*sizeof(*image1));
	unsigned char * image1_1D = (unsigned char *)malloc(width*height*sizeof(unsigned char));
	for(i=0;i<width;i++) {	image1[i] = &image1_1D[i*height];	}
	
	// Create space for image2
	if (messages == 2) { printf("### Allocating space for image2.\n"); }
	unsigned char ** image2 = (unsigned char **)malloc(width*sizeof(*image2));
	unsigned char * image2_1D = (unsigned char *)malloc(width*height*sizeof(unsigned char));
	for(i=0;i<width;i++) { image2[i] = &image2_1D[i*height];	}
	
	// Create space for image3
	if (messages == 2) { printf("### Allocating space for image3.\n"); }
	unsigned char ** image3 = (unsigned char **)malloc(width*sizeof(*image3));
	unsigned char * image3_1D = (unsigned char *)malloc(width*height*sizeof(unsigned char));
	for(i=0;i<width;i++) { image3[i] = &image3_1D[i*height];	}
	
	// Create space for projection vectors
	if (messages == 2) { printf("### Allocating space for projection vectors.\n"); fflush(stdout); }
	int * Xvector = (int *)malloc(width*sizeof(int));
	int * Yvector = (int *)malloc(height*sizeof(int));
	
	// Create coordinate arrays
	if (messages == 2) { printf("### Allocating space for coordinate arrays.\n"); fflush(stdout); }
	int Xcoordinates[XVECTORS]; for(i=0;i<XVECTORS;i++) { Xcoordinates[i] = 0; }
	int Ycoordinates[YVECTORS]; for(i=0;i<YVECTORS;i++) { Ycoordinates[i] = 0; }
	
	//########################################################################
	//### PART1: Histogramming (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART1: Histogramming.\n"); fflush(stdout); }
	
	#pragma species kernel 0:height-1,0:width-1|element -> 0:255|shared
	for (h=0;h<height;h++) {
		for (w=0;w<width;w++) {
			hist[image0[h][w]] = hist[image0[h][w]] + 1;
		}
	}
	#pragma species endkernel histogram
	
	//########################################################################
	//### Between class variance (CPU)
	//########################################################################
	if (messages == 2) { printf("### Create a between class variance table.\n"); fflush(stdout); }
	CPU_BCV(hist, BCVtable, width*height);

	//########################################################################
	//### PART2: Search for the maximum (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART2: Search for the maximum value.\n"); fflush(stdout); }
	float maximum[1];
	maximum[0] = 10;
	int length = 256;
	
	//#pragma species kernel 0:255|element -> 0:0|shared
	for (i=0;i<length;i++) {
		maximum[0] = (BCVtable[i] > maximum[0]) ? BCVtable[i] : maximum[0];
	}
	//#pragma species endkernel maximum_1
	
	if (messages == 2) { printf("### Maximum is %.3lf.\n",maximum[0]); fflush(stdout); }
	
	//########################################################################
	//### PART3: Search for the maximum - larger synthetic example (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART3: Search for the maximum value (synthetic example).\n"); fflush(stdout); }
	int vector_size = 2097152; // 2048x1024
	float* synthetic_vector = (float*)malloc(sizeof(float)*vector_size);
	srand(time(NULL));for (i=0;i<vector_size;i++) {	synthetic_vector[i] = (rand() % 7777777) / 1000.0; }
	float result[1];
	result[0] = 0;
	
	//#pragma species kernel 0:2097151|element -> 0:0|shared
	for (i=0;i<vector_size;i++) {
		result[0] = (synthetic_vector[i] > result[0]) ? synthetic_vector[i] : result[0];
	}
	//#pragma species endkernel maximum_2
	
	if (messages == 2) { printf("### Maximum is %.3lf.\n",result[0]); fflush(stdout); }
	
	//########################################################################
	//### Search for the index of the maximum (CPU)
	//########################################################################
	if (messages == 2) { printf("### Search for the index of the maximum value.\n"); fflush(stdout); }
	for (i=0;i<256;i++) {
		if (BCVtable[i] == maximum[0]) {
			threshold = i;
			break;
		}
	}
	
	//########################################################################
	//### PART4: Binarization (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART4: Binarization with treshold at %d.\n",threshold); fflush(stdout); }
	
	#pragma species kernel 0:height-1,0:width-1|element -> 0:height-1,0:width-1|element
	for (h=0;h<height;h++) {
		for (w=0;w<width;w++) {
			if (image0[h][w] > threshold) { image1[h][w] = 1; }
			else {                          image1[h][w] = 0; }
		}
	}
	#pragma species endkernel threshold
	
	//########################################################################
	//### PART5: Erosion 7x7 (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART5: Perform the erode kernel.\n"); fflush(stdout); }
	
	int condition;
	#pragma species kernel 7:height-8,7:width-8|neighbourhood(-3:3,-3:3) -> 0:height-1,0:width-1|element
	for (h=0;h<height;h++) {
		for (w=0;w<width;w++) {
			if (w >= 7 && h >= 7 && w <= width-7 && h <= height-7) {
				condition = 1;
				for(a=-3;a<=3;a++) {
					condition = condition
											* image1[(h-3)][(w+a)]
											* image1[(h-2)][(w+a)]
											* image1[(h-1)][(w+a)]
											* image1[(h+0)][(w+a)]
											* image1[(h+1)][(w+a)]
											* image1[(h+2)][(w+a)]
											* image1[(h+3)][(w+a)]
					;
				}
				if (condition == 1) { image2[h][w] = 255; }
				else {                image2[h][w] = 0; }
			}
			else {
				image2[h][w] = 0;
			}
		}
	}
	#pragma species endkernel erosion
	
	//########################################################################
	//### PART6: 1D erosion(7) synthetic example (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART6: Perform the erode kernel (1D - synthetic).\n"); fflush(stdout); }
	int vector_size2 = 2097152; // 2048x1024
	int* vector2a = (int*)malloc(sizeof(int)*vector_size2);
	int* vector2b = (int*)malloc(sizeof(int)*vector_size2);
	srand(time(NULL));
	for (i=0;i<vector_size2;i++) {
		if (rand()%15 > 1) { vector2a[i] = 1; }
		else              { vector2a[i] = 0; }
	}
	
	//#pragma species kernel 0:2097151|neighbourhood(-3:3) -> 0:2097151|element
	for (i=0;i<vector_size2;i++) {
		if (i >= 7 && i <= vector_size2-7) {
			condition = 1;
			for(a=-3;a<=3;a++) {
				condition = condition * vector2a[i+a];
			}
			if (condition == 1) { vector2b[i] = 255; }
			else {                vector2b[i] = 0; }
		}
		else {
			vector2b[i] = 0;
		}
	}
	//#pragma species endkernel erosion1d
	
	// Compute a gold reference
	int gold = 0;
	int gold_condition = 1;
	for(a=-3;a<=3;a++) { gold_condition = gold_condition * vector2a[10+a]; }
	if (gold_condition == 1) { gold = 255; }
	if (messages == 2) { printf("### Result at index 10 is %d and should be %d.\n",vector2b[10],gold); fflush(stdout); }
	
	//########################################################################
	//### PART7: Y-projection (accelerated)
	//########################################################################	
	if (messages >= 1) { printf("### PART7: Starting the Y-projection algorithm.\n"); fflush(stdout); }
	
	int result_yp;
	#pragma species kernel 0:height-1,0:width-1|chunk(0:height-1,0:0) -> 0:width-1|element
	for (w=0;w<width;w++) {
		result_yp = 0;
		for (h=0;h<height;h++) {
			if (image2[h][w] == 255) {
				result_yp = 255;
			}
		}
		Yvector[w] = result_yp;
	}
	#pragma species endkernel y_projection
	
	//########################################################################
	//### PART8: X-projection (accelerated)
	//########################################################################
	if (messages >= 1) { printf("### PART8: Starting the X-projection algorithm.\n"); fflush(stdout); }
	
	int result_xp;
	#pragma species kernel 0:height-1,0:width-1|chunk(0:0,0:width-1) -> 0:height-1|element
	for (h=0;h<height;h++) {
		result_xp = 0;
		for (w=0;w<width;w++) {
			if (image2[h][w] == 255) {
				result_xp = 255;
			}
		}
		Xvector[h] = result_xp;
	}
	#pragma species endkernel x_projection
	
	//########################################################################
	//### Search for the centers of the projection vectors (CPU)
	//########################################################################	
	if (messages == 2) { printf("### Search for X- and Y-projection vectors.\n"); fflush(stdout); }
	CPU_FindCenters(Xvector, Xcoordinates, width);
	CPU_FindCenters(Yvector, Ycoordinates, height);
		
	//########################################################################
	//### Visualize, save to disk and finalize the program
	//########################################################################
	CPU_Visualize(image0, Xcoordinates, Ycoordinates, image3, width, height);
	SaveBMPFile(image1, "output1.bmp", width, height);
	SaveBMPFile(image2, "output2.bmp", width, height);
	SaveBMPFile(image3, "output3.bmp", width, height);
	free(image0);
	free(image1);
	free(image1_1D);
	free(image2);
	free(image2_1D);
	free(image3);
	free(image3_1D);
	free(Xvector);
	free(Yvector);
	free(BCVtable);
	if (messages == 2) { printf("### End of program\n"); fflush(stdout); }
	return 0;
}

//########################################################################
//### Structures used in the BMP functions
//########################################################################

typedef struct {
	int size;
	int reserved;
	int offset;
} BMPHeader;
typedef struct {
	int size;
	int width;
	int height;
	int planesBitsPerPixel;
	int compression;
	int imageSize;
	int xPelsPerMeter;
	int yPelsPerMeter;
	int clrUsed;
	int clrImportant;
} BMPInfoHeader;

//########################################################################
//### Function to save BMP data to a file
//########################################################################

void SaveBMPFile(unsigned char ** image, const char * outputdestination, int width, int height)
{
	// Variable declarations
	int x,y,j;
	FILE *fd_out;
  unsigned long ulBitmapSize = (height * width * 3)+54; 
  char ucaBitmapSize[4];
  ucaBitmapSize[3]= (ulBitmapSize & 0xFF000000) >> 24;
  ucaBitmapSize[2]= (ulBitmapSize & 0x00FF0000) >> 16;
  ucaBitmapSize[1]= (ulBitmapSize & 0x0000FF00) >> 8;
  ucaBitmapSize[0]= (ulBitmapSize & 0x000000FF);
  	
	// Load output file
	fd_out = fopen(outputdestination, "wb");
 		
	// Write BMP header
	fprintf(fd_out,"%c%c%c%c%c%c%c%c%c%c", 66, 77, ucaBitmapSize[0], ucaBitmapSize[1], ucaBitmapSize[2], ucaBitmapSize[3], 0, 0, 0, 0); 
	fprintf(fd_out,"%c%c%c%c%c%c%c%c%c%c", 54, 0, 0, 0, 40, 0 , 0, 0, (width & 0x00FF), (width & 0xFF00)>>8); 
	fprintf(fd_out,"%c%c%c%c%c%c%c%c%c%c", 0, 0, (height & 0x00FF), (height & 0xFF00) >> 8, 0, 0, 1, 0, 24, 0); 
	fprintf(fd_out,"%c%c%c%c%c%c%c%c%c%c", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); 
	fprintf(fd_out,"%c%c%c%c%c%c%c%c%c%c", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	fprintf(fd_out,"%c%c%c%c", 0, 0 ,0, 0);

	// Save RGB data to output file	
	for(y=0;y<height;y++) {
		for(x=0;x<width;x++)	{
			fputc(image[x][y],fd_out);
			fputc(image[x][y],fd_out);
			fputc(image[x][y],fd_out);
		}
		int over = width%4;
		if (over != 0) {
			for(j=0;j<over;j++) {
				fputc(0,fd_out);
			}
		}
	}

	// Clean up
	fclose(fd_out);
}

//########################################################################
//### Function to load BMP data from disk
//########################################################################

unsigned char ** LoadBMPFile(int *width, int *height)
{
	// Variable declarations
	short type;
	int temp;
	BMPHeader hdr;
	BMPInfoHeader infoHdr;
	FILE *fd;
	int i, y, x;
	
	// Open the file stream
	fd = fopen("../../../input.bmp","rb");

	// Open the file and scan the contents
	if(!(fd)) { printf("***BMP load error: file access denied***\n"); exit(0);	}
	temp = fread(&type, sizeof(short), 1, fd);
	temp = fread(&hdr, sizeof(hdr), 1, fd);
	if(type != 0x4D42) { printf("***BMP load error: bad file format***\n"); exit(0); }
	temp = fread(&infoHdr, sizeof(infoHdr), 1, fd);
	if((infoHdr.planesBitsPerPixel>>16) != 24) { printf("***BMP load error: invalid color depth (%d)*** \n",(infoHdr.planesBitsPerPixel>>16)); exit(0); }
	if(infoHdr.compression) { printf("***BMP load error: compressed image***\n"); exit(0); }
	(*width)  = infoHdr.width;
	(*height) = infoHdr.height;

	// Allocate memory to store the BMP's contents
	unsigned char ** image = (unsigned char **)malloc((*width) * sizeof(*image));
	unsigned char * image_1D = (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
	for(i=0; i<(*width); i++) {
		image[i] = &image_1D[i*(*height)];
	}

	// Read the BMP file and store the contents
	fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);
	for(y = 0; y < (*height); y++) {
		for(x = 0; x < (*width); x++)	{
			image[x][y] = ((int)fgetc(fd));
			fgetc(fd);
			fgetc(fd);
		}
		int over = (4 - ((*width)*3) % 4) % 4;
		if (over != 0) {
			for(x = 0; x < over; x++)	{
				fgetc(fd);
			}
		}
	}

	// Exit the function and clean-up
	if(ferror(fd)) {
		printf("***Unknown BMP load error.***\n");
		free(image[0]);
		free(image);
		exit(0);
	}
	fclose(fd);
	return image;
}

//########################################################################
//### Find the center of a projection vector (using a state machine)
//########################################################################		
void CPU_FindCenters(int* vector, int *coordinates, int size) {
	int s;
	int state = 0;
	int count = 0;
	int coordinate = 0;
	for (s=0;s<size;s++) {
		if (state == 0) {                           // Last thing I found was a zero
			if (vector[s] == 255) {                  // I found a 255 now
				state = 1;
				count = 0;
			}
		}		
		if (state == 1) {                           // Last thing I found was 255
			if (vector[s] == 0) {                    	// I found a zero now
				state = 0;
				if (count > 4) {                        // To filter out noise
					coordinates[coordinate] = s-(count/2);
					coordinate++;
				}
			}	
			else {                                    // I found a 255 again
				count++;
			}
		}
	}
}

//########################################################################
//### CPU kernel to visualize the results
//########################################################################
void CPU_Visualize(unsigned char** image0, int* Xcoordinates, int* Ycoordinates, unsigned char **image3, int width, int height) {

	// Loop variables
	int h, w, x, y;

	// Copy the whole image
	for (h=0;h<height;h++) {
		for (w=0;w<width;w++) {
			unsigned char value = image0[h][w];
			image3[h][w] = value;
		}
	}
	
	// Replace the centers with white pixels
	for (x=0;x<XVECTORS;x++) {
		for (y=0;y<YVECTORS;y++) {
			image3[Xcoordinates[x]][Ycoordinates[y]] = 255;
		}
	}
}

//########################################################################
//### CPU kernel function for between class variance (BCV), part of Otsu thresholding
//########################################################################		
void CPU_BCV(int *histogram, float *BCVtable, int size) {
	int i;
	
	// Initialize the BCV table to zero
	for (i=0;i<256;i++) {
		BCVtable[i] = 0;
	}
	
	// Pre-calculated the total of the weigthed sums
	int wsumtotal = 0;
	for (i=0;i<256;i++) {
		wsumtotal = wsumtotal + i*histogram[i];
	}
	
	// Set the initial values
	int sumb = 0;
	int sumf = size;	
	int wsumb = 0;
	int wsumf = wsumtotal;
	
	float wb;
	float wf;
	float meanb;
	float meanf;

	// Iterate over all possible threshold values
	for (i=0;i<256;i++)	{

		// Update the weighted sums
		wsumb = wsumb + i*histogram[i];
		wsumf = wsumtotal - wsumb;
		
		// Calculate the necessary components
		wb = sumb / (float)size;
		wf = sumf / (float)size;
		meanb = wsumb / (float)sumb;
		meanf = wsumf / (float)sumf;
	
		// Stop if the sum of foreground is equal to zero
		if (sumf == 0) { break; }
		
		// Output the BCV value
		BCVtable[i] = wb*wf*(meanb-meanf)*(meanb-meanf);		
		
		// If the sum of the background was equal to zero, BCV table will be NaN and must be set to zero
		if (sumb == 0) { BCVtable[i] = 0; }		
		
		// Update the sum of the background (all darker pixels compared to the current pixel) and foreground pixels (the rest)
		sumb = sumb + histogram[i];
		sumf = size - sumb;
	}
}

//########################################################################
