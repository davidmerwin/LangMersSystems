#include <stdio.h>
#include <stdlib.h>

// Basic interface with device's hardware
void capture_image() {
  system("raspistill -o image.jpg");
}

void capture_audio() {
  system("arecord -D hw:2,0 -f cd -c1 -r 48000 -d 5 -t wav audio.wav");
}
