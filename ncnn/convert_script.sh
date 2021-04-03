rm -rf *.param *.bin *.onnx
python3 torch2onxx.py
echo "Torch to onnx ok!!!!!!!!!!!!!!!!!"
#python3 -m onnxsim  model.onnx model_sim.onnx
#echo "Onnx to onnx simplified ok !!!!!!!!!!!!!"
./onnx2ncnn model_sim.onnx model.param model.bin
echo "onnx to ncnn ok !!!!!!!!!!!!!!!!!!!!"
