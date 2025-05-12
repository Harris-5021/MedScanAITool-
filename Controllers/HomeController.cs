using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace YourProject.Controllers
{
    public class HomeController : Controller
    {
        private readonly IWebHostEnvironment _environment;

        public HomeController(IWebHostEnvironment environment)
        {
            _environment = environment;
        }

      public IActionResult Index()
{
    return View(); // This will render Views/Home/Index.cshtml
}
 public IActionResult Privacy()
        {
            return View();
        }
        public IActionResult UserDashboard()
        {
            ViewBag.Message = "Model loaded successfully. You can now analyze X-ray images.";
            ViewBag.MessageType = "alert-success";
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> UploadXRay(IFormFile xrayImage)
        {
            if (xrayImage == null || xrayImage.Length == 0)
            {
                ViewBag.Message = "Please upload a valid X-ray image.";
                ViewBag.MessageType = "alert-danger";
                return View("UserDashboard");
            }

            // Save the uploaded image
            var uploadsPath = Path.Combine(_environment.WebRootPath, "uploads");
            if (!Directory.Exists(uploadsPath))
                Directory.CreateDirectory(uploadsPath);

            var imagePath = Path.Combine(uploadsPath, xrayImage.FileName);
            using (var stream = new FileStream(imagePath, FileMode.Create))
            {
                await xrayImage.CopyToAsync(stream);
            }

            // Call Python script for prediction and Grad-CAM
            var pythonScriptPath = Path.Combine(_environment.ContentRootPath, "condition_api", "gradcam_utils.py");
            var modelPath = Path.Combine(_environment.ContentRootPath, "condition_api", "models", "multilabel_vit_model.pth");
            var outputPath = Path.Combine(uploadsPath, "output.txt");
            var gradcamOutputPath = Path.Combine(uploadsPath, "gradcam_");

            var processInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{pythonScriptPath}\" \"{imagePath}\" \"{modelPath}\" \"{outputPath}\" \"{gradcamOutputPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            string result, error;
            using (var process = Process.Start(processInfo))
            {
                result = process.StandardOutput.ReadToEnd();
                error = process.StandardError.ReadToEnd();
                process.WaitForExit();
            }

            if (!string.IsNullOrEmpty(error))
            {
                ViewBag.Message = $"Error during analysis: {error}";
                ViewBag.MessageType = "alert-danger";
                return View("UserDashboard");
            }

            // Read predictions from output file
            var predictions = System.IO.File.ReadAllText(outputPath);
            ViewBag.Message = $"Analysis Results: {predictions}";
            ViewBag.MessageType = "alert-info";
            ViewBag.ImagePath = $"/uploads/{xrayImage.FileName}";

            // Pass Grad-CAM image paths
            ViewBag.GradCamPaths = new Dictionary<string, string>();
            foreach (var condition in new[] { "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral", "Pneumothorax", "Tuberculosis" })
            {
                var gradcamPath = $"{gradcamOutputPath}{condition}.png";
                if (System.IO.File.Exists(gradcamPath))
                {
                    ViewBag.GradCamPaths[condition] = $"/uploads/gradcam_{condition}.png";
                }
            }

            return View("Index");
        }
    }
}