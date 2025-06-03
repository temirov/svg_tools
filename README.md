# SVG Tools

A collection of tools that facilitates working with SVGs

## Base64

```shell
uv run --with pillow --python 3 main.py \
    --size 32 \
    assets/Calendar\ Icon\ with\ Checkmark\ on\ Blue.png
```

## Image Vectorization

```shell
uv run --with opencv-python --with numpy --with Pillow python vectorize_image.py
```

## Text vectorization

```shell
uv run --with fonttools --with svgpathtools --with svgwrite python text_to_svg.py \
       "Marko Polo Research Lab" --font assets/GreatVibes-Regular.ttf --out title.svg
```

## Silhouette

```shell
uv run --with fonttools --with opencv-python --with numpy python image_to_silhouette.py \
       --input-file assets/marko_polo_portrait.jpg \
       --output-file mp_silhouette.png
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
