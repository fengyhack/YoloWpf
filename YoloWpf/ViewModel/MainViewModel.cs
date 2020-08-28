using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Command;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Threading;
using System.Threading.Tasks;
using System.Web.UI.WebControls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace YoloWpf.ViewModel
{
    public class MainViewModel : ViewModelBase
    {
        private const string BASE_FOLDER = @"E:\Workspace\Datasets\";

        private Net darkNet;

        private string[] classNames;
        private bool modelChanged = false;

        private const float confidenceThreshold = 0.5f;
        private const float nmsThreshold = 0.3f;

        private Scalar[] randomColors;
        private Mat sourceImage;
        private Mat[] detectionOutputs;

        /// <summary>
        /// Initializes a new instance of the MainViewModel class.
        /// </summary>
        public MainViewModel()
        {
            modelChanged = true;
            classNames = new string[1];
            randomColors = Enumerable.Repeat(false, 80).Select(x => Scalar.RandomColor()).ToArray();
        }

        private bool _DisplaySetupWidget = true;
        public bool DisplaySetupWidget
        {
            get
            {
                return _DisplaySetupWidget;
            }
            set
            {
                if (_DisplaySetupWidget != value)
                {
                    _DisplaySetupWidget = value;
                    RaisePropertyChanged();
                }
            }
        }

        private bool _IsReadyForChanges = true;
        public bool IsReadyForChanges
        {
            get
            {
                return _IsReadyForChanges;
            }
            set
            {
                if (_IsReadyForChanges != value)
                {
                    _IsReadyForChanges = value;
                    RaisePropertyChanged();
                }
            }
        }

        private bool _IsReadyForDetetction = false;
        public bool IsReadyForDetetction
        {
            get
            {
                return _IsReadyForDetetction;
            }
            set
            {
                if (_IsReadyForDetetction != value)
                {
                    _IsReadyForDetetction = value;
                    RaisePropertyChanged();
                }
            }
        }

        private string _ConfigFileName = $"{BASE_FOLDER}yolov3.cfg";
        public string ConfigFileName
        {
            get
            {
                return Path.GetFileName(_ConfigFileName);
            }
            set
            {
                if (_ConfigFileName != value)
                {
                    modelChanged = true;
                    _ConfigFileName = value;
                    RaisePropertyChanged();
                }
            }
        }

        private string _ClassesFileName = $"{BASE_FOLDER}coco.names";
        public string ClassesFileName
        {
            get
            {
                return Path.GetFileName(_ClassesFileName);
            }
            set
            {
                if (_ClassesFileName != value)
                {
                    modelChanged = true;
                    _ClassesFileName = value;
                    RaisePropertyChanged();
                }
            }
        }

        private string _WeightsFileName = $"{BASE_FOLDER}yolov3.weights";
        public string WeightsFileName
        {
            get
            {
                return Path.GetFileName(_WeightsFileName);
            }
            set
            {
                if (_WeightsFileName != value)
                {
                    modelChanged = true;
                    _WeightsFileName = value;
                    RaisePropertyChanged();
                }
            }
        }

        private string _ImageFileName;
        public string ImageFileName
        {
            get
            {
                return _ImageFileName;
            }
            set
            {
                if (_ImageFileName != value)
                {
                    _ImageFileName = value;
                    if (File.Exists(_ImageFileName))
                    {
                        ImageSource = new BitmapImage(new Uri(_ImageFileName));
                    }
                }
            }
        }

        private BitmapSource _ImageSource;
        public BitmapSource ImageSource
        {
            get
            {
                return _ImageSource;
            }
            set
            {
                if (_ImageSource != value)
                {
                    _ImageSource = value;
                    RaisePropertyChanged();
                }
            }
        }

        private ICommand _SetupModelCommand;
        public ICommand SetupModelCommand
        {
            get
            {
                return _SetupModelCommand ?? (_SetupModelCommand = new RelayCommand(
                    () =>
                    {
                        DisplaySetupWidget = true;
                    })
                );
            }
        }

        private ICommand _SelectFileCommand;
        public ICommand SelectFileCommand
        {
            get
            {
                return _SelectFileCommand ?? (_SelectFileCommand = new RelayCommand<string>(
                    (str) =>
                    {
                        switch(str)
                        {
                            case "Config":
                                ConfigFileName = SelectFile("Config Files|*.cfg", _ConfigFileName);
                                break;
                            case "Classes":
                                ClassesFileName = SelectFile("Classes Files|*.names", _ClassesFileName);
                                break;
                            case "Weights":
                                WeightsFileName = SelectFile("Weights Files|*.weights", _WeightsFileName);
                                break;
                            case "Image":
                                ImageFileName = SelectFile("Image Files|*.jpg;*.png|All Files|*.*", _ImageFileName);
                                break;
                        }

                        IsReadyForDetetction = CheckReadyForDetection();
                        if(IsReadyForDetetction && modelChanged)
                        {
                            darkNet = CvDnn.ReadNetFromDarknet(_ConfigFileName, _WeightsFileName);
                            darkNet.SetPreferableBackend(Net.Backend.OPENCV);
                            darkNet.SetPreferableTarget(Net.Target.CPU);
                            classNames = File.ReadAllLines(_ClassesFileName);
                            modelChanged = false;                            
                        }

                        DisplaySetupWidget = !IsReadyForDetetction;
                    })
                );
            }
        }

        private ICommand _RandomBBoxColor;
        public ICommand RandomBBoxColor
        {
            get
            {
                return _RandomBBoxColor ?? (_RandomBBoxColor = new RelayCommand(
                    () =>
                    {
                        randomColors = Enumerable.Repeat(false, 80).Select(x => Scalar.RandomColor()).ToArray();

                        if (sourceImage != null && detectionOutputs != null && detectionOutputs.Any())
                        {
                            var isReady = IsReadyForDetetction;
                            IsReadyForChanges = false;
                            IsReadyForDetetction = false;
                            Task.Run(() =>
                            {
                                DrawResults(detectionOutputs, sourceImage, confidenceThreshold, nmsThreshold);

                                App.Current.Dispatcher.Invoke(() =>
                                {
                                    ImageSource = Imaging.CreateBitmapSourceFromHBitmap(
                                        sourceImage.ToBitmap().GetHbitmap(),
                                        IntPtr.Zero, System.Windows.Int32Rect.Empty,
                                        BitmapSizeOptions.FromEmptyOptions());
                                });

                                IsReadyForChanges = true;
                                IsReadyForDetetction = isReady;
                            });                            
                        }
                    }));
            }
        }

        private ICommand _StartDetectionCommand;
        public ICommand StartDetectionCommand
        {
            get
            {
                return _StartDetectionCommand ?? (_StartDetectionCommand = new RelayCommand(
                    () =>
                    {
                        DisplaySetupWidget = false;
                        IsReadyForDetetction = false;
                        IsReadyForChanges = false;
                        Task.Run(() =>
                        {
                            PerfomrDetection(darkNet, _ImageFileName);
                            IsReadyForChanges = true;
                            IsReadyForDetetction = true;
                        });
                    })
                );
            }
        }

        private bool CheckReadyForDetection()
        {
            return (File.Exists(_ImageFileName) && File.Exists(_ConfigFileName) && File.Exists(_ClassesFileName) && File.Exists(_WeightsFileName));
        }

        private string SelectFile(string filter, string fallback = "")
        {
            var ofd = new OpenFileDialog();
            ofd.Filter = filter;
            ofd.Multiselect = false;
            if (ofd.ShowDialog() ?? true)
            {
                return ofd.FileName;
            }
            else
            {
                return fallback;
            }
        }

        private void PerfomrDetection(Net net, string imageFileName)
        {
            sourceImage = new Mat(imageFileName);
            const int MAX_WIDTH = 1200;
            const int MAX_HEIGHT = 800;
            if (sourceImage.Width > MAX_WIDTH || sourceImage.Height > MAX_HEIGHT)
            {
                double fx = (double)MAX_WIDTH / sourceImage.Width;
                double fy = (double)MAX_HEIGHT / sourceImage.Height;
                double ff = fx < fy ? fx : fy;
                sourceImage = sourceImage.Resize(new Size(0, 0), ff, ff);
            }

            App.Current.Dispatcher.Invoke(() =>
            {
                ImageSource = Imaging.CreateBitmapSourceFromHBitmap(
                    sourceImage.ToBitmap().GetHbitmap(),
                    IntPtr.Zero, System.Windows.Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
            });

            //setting blob, size can be:320/416/608
            //opencv blob setting can check here https://github.com/opencv/opencv/tree/master/samples/dnn#object-detection
            var blob = CvDnn.BlobFromImage(sourceImage, 1.0 / 255, new Size(416, 416), new Scalar(), true, false);
            net.SetInput(blob);
            var outputNames = net.GetUnconnectedOutLayersNames();
            //create mats for output layer
            detectionOutputs = outputNames.Select(_ => new Mat()).ToArray();
            net.Forward(detectionOutputs, outputNames);

            DrawResults(detectionOutputs, sourceImage, confidenceThreshold, nmsThreshold);

            App.Current.Dispatcher.Invoke(() =>
            {
                ImageSource = Imaging.CreateBitmapSourceFromHBitmap(
                    sourceImage.ToBitmap().GetHbitmap(),
                    IntPtr.Zero, System.Windows.Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
            });
        }

        /// <summary>
        /// Get result form all output
        /// </summary>
        /// <param name="outputs"></param>
        /// <param name="image"></param>
        /// <param name="confidenceThreshold"></param>
        /// <param name="nmsThreshold">threshold for nms</param>
        /// <param name="nms">Enable Non-maximum suppression or not</param>
        private void DrawResults(IEnumerable<Mat> outputs, Mat image, float confidenceThreshold, float nmsThreshold, bool nms = true)
        {
            //for nms
            var classIds = new List<int>();
            var confidences = new List<float>();
            var probabilities = new List<float>();
            var boxes = new List<Rect2d>();

            var w = image.Width;
            var h = image.Height;
            /*
             YOLO3 COCO trainval output
             0 1 : center                    
             2 3 : w/h
             4 : confidence                  
             5 ~ 84 : class probability 
            */
            const int prefix = 5;   //skip 0~4

            foreach (var output in outputs)
            {
                for (var i = 0; i < output.Rows; i++)
                {
                    var confidence = output.At<float>(i, 4);
                    if (confidence > confidenceThreshold)
                    {
                        //get classes probability
                        Cv2.MinMaxLoc(output.Row(i).ColRange(prefix, output.Cols), out _, out double max);
                        var classes = (int)max;
                        var probability = output.At<float>(i, classes + prefix);

                        if (probability > confidenceThreshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            var centerX = output.At<float>(i, 0) * w;
                            var centerY = output.At<float>(i, 1) * h;
                            var width = output.At<float>(i, 2) * w;
                            var height = output.At<float>(i, 3) * h;

                            if (!nms)
                            {
                                // draw result (if don't use NMSBoxes)
                                DrawResult(image, classes, confidence, probability, centerX, centerY, width, height);
                                continue;
                            }

                            //put data to list for NMSBoxes
                            classIds.Add(classes);
                            confidences.Add(confidence);
                            probabilities.Add(probability);
                            boxes.Add(new Rect2d(centerX, centerY, width, height));
                        }
                    }
                }
            }

            if (!nms) return;

            //using non-maximum suppression to reduce overlapping low confidence box
            CvDnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, out int[] indices);

            foreach (var i in indices)
            {
                var box = boxes[i];
                DrawResult(image, classIds[i], confidences[i], probabilities[i], box.X, box.Y, box.Width, box.Height);
            }
        }

        /// <summary>
        /// Draw result to image
        /// </summary>
        /// <param name="image"></param>
        /// <param name="classes"></param>
        /// <param name="confidence"></param>
        /// <param name="probability"></param>
        /// <param name="centerX"></param>
        /// <param name="centerY"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        private void DrawResult(Mat image, int classes, float confidence, float probability, double centerX, double centerY, double width, double height)
        {
            //label formating
            var label = $"{classNames[classes]} {probability * 100:0.00}%";
            var x1 = (centerX - width / 2) < 0 ? 0 : centerX - width / 2; //avoid left side over edge
            //draw result
            image.Rectangle(new Point(x1, centerY - height / 2), new Point(centerX + width / 2, centerY + height / 2), randomColors[classes], 2);
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);
            Cv2.Rectangle(image, new Rect(new Point(x1, centerY - height / 2 - textSize.Height - baseline),
                new Size(textSize.Width, textSize.Height + baseline)), randomColors[classes], Cv2.FILLED);
            var textColor = Cv2.Mean(randomColors[classes]).Val0 < 70 ? Scalar.White : Scalar.Black;
            Cv2.PutText(image, label, new Point(x1, centerY - height / 2 - baseline), HersheyFonts.HersheyTriplex, 0.5, textColor);
        }

    }
}