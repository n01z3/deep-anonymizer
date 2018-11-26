# Item swap
Consist of two part: swapping face and swaping background.
## Using
### For initialization:
```
- git clone https://github.com/EvgenyKashin/ItemSwap.git
- make docker-build
- make load-weights
```

***Important:*** run docker under sudo. Also may be required
```sudo chmod -R 777 ./*``` (sorry)

### For training:
TODO: write instruction

### For face and background swap:
```
- make convert-face-video
- make convert-face-image
- make convert-background-image
```
For parameters see Makefile

***Important:*** there is ```loadSize``` parameter in ```make convert-background-image``` for changing image size.
