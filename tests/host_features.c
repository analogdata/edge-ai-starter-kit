// Host-side wrapper: reads KWS_CLIP_SAMPLES int16 from argv[1], writes float32 features to argv[2].
#include <stdio.h>
#include "../firmware/main/features.h"

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    static int16_t in[KWS_CLIP_SAMPLES];
    static float out[KWS_NUM_FRAMES * KWS_NUM_MEL];
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(in, sizeof(int16_t), KWS_CLIP_SAMPLES, f) != KWS_CLIP_SAMPLES) return 3;
    fclose(f);
    features_init();
    features_compute(in, out);
    f = fopen(argv[2], "wb");
    fwrite(out, sizeof(float), KWS_NUM_FRAMES * KWS_NUM_MEL, f);
    fclose(f);
    return 0;
}
