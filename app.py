#!/usr/bin/env python3

from pathlib import Path
import argparse
import logging
import json
import cv2
import numpy as np
import math

import fastmot
from fastmot.utils import ConfigDecoder, Profiler

from Transform import PixelMapper

def getAoA(px, py):
    quad_coords = {
        "lonlat": np.array([
            [3.22, 0.35], # Third lampost top right
            [1.38, 2.66], # Corner of white rumble strip top left
            [9.76, 12.88], # Corner of rectangular road marking bottom left
            #[25.76, 6.08] # Corner of dashed line bottom right
            [25.66, 3.68] # Corner of dashed line bottom right
        ]),
        "pixel": np.array([
            [127, 661], # Third lampost top right
            [1269, 688], # Corner of white rumble strip top left
            [1110, 113], # Corner of rectangular road marking bottom left
            #[173, 74] # Corner of dashed line bottom right
            [21, 84] # Corner of dashed line bottom right
        ])
    }

    pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    rloc = pm.pixel_to_lonlat((px, py))
    aoa = math.atan2(rloc[0][1], rloc[0][0])*180.0/3.14

    return aoa

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input_uri', metavar="URI", required=True, help=
                        'URI to input stream\n'
                        '1) image sequence (e.g. img_%%06d.jpg)\n'
                        '2) video file (e.g. video.mp4)\n'
                        '3) MIPI CSI camera (e.g. csi://0)\n'
                        '4) USB/V4L2 camera (e.g. /dev/video0)\n'
                        '5) RTSP stream (rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                        '6) HTTP stream (http://<user>:<password>@<ip>:<port>/<path>)\n')
    parser.add_argument('-c', '--config', metavar="FILE",
                        default=Path(__file__).parent / 'cfg' / 'mot.json',
                        help='path to configuration JSON file')
    parser.add_argument('-o', '--output_uri', metavar="URI",
                        help='URI to output video (e.g. output.mp4)')
    parser.add_argument('-l', '--log', metavar="FILE",
                        help='output a MOT Challenge format log (e.g. eval/results/mot17-04.txt)')
    parser.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    parser.add_argument('-g', '--gui', action='store_true', help='enable display')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output for debugging')
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder)

    mot = None
    log = None
    stream = fastmot.VideoIO(config['resize_to'], config['video_io'], args.input_uri, args.output_uri)

    if args.mot:
        draw = args.gui or args.output_uri is not None
        mot = fastmot.MOT(config['resize_to'], config['mot'], draw=draw, verbose=args.verbose)
        mot.reset(stream.cap_dt)
        if args.log is not None:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.log, 'w')
    if args.gui:
        cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    logger.info('Starting video capture...')
    stream.start_capture()
    locs = []
    try:
        with Profiler('app') as prof:
            while not args.gui or cv2.getWindowProperty("Video", 0) >= 0:
                frame = stream.read()
                if frame is None:
                    break

                if args.mot:
                    mot.step(frame)
                    if log is not None:
                        for track in mot.visible_tracks:
                            row = [mot.frame_count, track.trk_id]
                            row.extend(track.tlbr)
                            #print (row)
                            px = round((float(row[4])-float(row[2]))/2+float(row[2]))
                            py = float(row[5])
                            locs.append(row)
                            aoa = getAoA(px, py)
                            print (aoa)
                            tl = track.tlbr[:2] / config['resize_to'] * stream.resolution
                            br = track.tlbr[2:] / config['resize_to'] * stream.resolution
                            w, h = br - tl + 1
                            log.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                      f'{w:.6f},{h:.6f},-1,-1,-1\n')

                if args.gui:
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if args.output_uri is not None:
                    stream.write(frame)
    finally:
        # clean up resources
        if log is not None:
            log.close()
        stream.release()
        cv2.destroyAllWindows()

    np.savetxt('pixel-locs.txt', np.array(locs), fmt='%s')

    if args.mot:
        # timing statistics
        avg_fps = round(mot.frame_count / prof.duration)
        logger.info('Average FPS: %d', avg_fps)
        mot.print_timing_info()


if __name__ == '__main__':
    main()
