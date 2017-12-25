/*

This is a demo of read pixel data from bmp file

*/

#include <Windows.h>
#include <stdio.h>

#define WMIMG "input.bmp"

int main()
{
	UINT8* frame_buffer;

	// Read bmp file header and info header
	BITMAPFILEHEADER header;
	BITMAPINFOHEADER info;
	FILE *bmpFile = fopen(WMIMG, "rb");
	if (!bmpFile) return -1;
	if (fread(&header, sizeof(BITMAPFILEHEADER), 1, bmpFile) != 1)
		return -1;
	if (fread(&info, sizeof(BITMAPINFOHEADER), 1, bmpFile) != 1)
		return -1;

	frame_buffer = (UINT8*)malloc(info.biHeight * info.biWidth * 4);
	memset(frame_buffer, 0, info.biHeight * info.biWidth * 4);

	// read rgba data
	if (info.biBitCount == 32) // 32 or 24 deep bmp
	{
		printf("biBitCount=%d\n", info.biBitCount);
		fread(frame_buffer, 4, info.biHeight * info.biWidth, bmpFile);
	}
	else
	{
		return -1;
	}

	return 0;
}