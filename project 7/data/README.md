# Data Directory

This directory contains sample images used by the Wardrobe.AI application.

## Structure

- `pant_samples/` - Contains sample pant/bottomwear images
- `tshirt_samples/` - Contains sample t-shirt/topwear images

## Adding More Images

### For Pants/Bottomwear:
1. Add new pant images to the `pant_samples/` folder
2. Supported formats: PNG, JPG, JPEG
3. Recommended resolution: 512x512 or higher
4. Images should have transparent backgrounds for best AR overlay results

### For T-shirts/Topwear:
1. Add new t-shirt images to the `tshirt_samples/` folder
2. Supported formats: PNG, JPG, JPEG
3. Recommended resolution: 512x512 or higher
4. Images should have transparent backgrounds for best AR overlay results

## Notes

- The application uses a vector database (Chroma) to find matching clothing items
- Images are processed and embedded for similarity search
- Current samples are just examples - add your own clothing catalog here
- Ensure image quality is good for better AI recognition and matching
