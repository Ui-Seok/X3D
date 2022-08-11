{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from hmdb51 import *\n",
    "from slowfast.utils.parser import load_config\n",
    "def main():\n",
    "    \"\"\"\n",
    "    Main function to spawn the train and test process.\n",
    "    \"\"\"\n",
    "    args = parse_args()\n",
    "    print(\"config files: {}\".format(args.cfg_files))\n",
    "    for path_to_config in args.cfg_files:\n",
    "        cfg = load_config(args, path_to_config)\n",
    "\n",
    "        # Perform training.\n",
    "        if cfg.TRAIN.ENABLE:\n",
    "            launch_job(cfg=cfg, init_method=args.init_method, func=train)\n",
    "\n",
    "        # Perform multi-clip testing.\n",
    "        if cfg.TEST.ENABLE:\n",
    "            launch_job(cfg=cfg, init_method=args.init_method, func=test)\n",
    "\n",
    "        # Perform model visualization.\n",
    "        if cfg.TENSORBOARD.ENABLE and (\n",
    "            cfg.TENSORBOARD.MODEL_VIS.ENABLE\n",
    "            or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE\n",
    "        ):\n",
    "            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)\n",
    "\n",
    "        # Run demo.\n",
    "        if cfg.DEMO.ENABLE:\n",
    "            demo(cfg)\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.8.13 ('X3D')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "585529c16dd72b585868480fed63d5dd1e3fe119d6d84462548113b73bdab698"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
