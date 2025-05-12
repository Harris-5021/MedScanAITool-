using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Authorization;
using System.Text.Json;
using System.Security.Claims;

namespace WebApplication1.Controllers
{
    [Authorize]
    public class DashboardController : Controller
    {
        private readonly IWebHostEnvironment _webHostEnvironment;
        private readonly IHttpClientFactory _clientFactory;

        public DashboardController(IWebHostEnvironment webHostEnvironment, IHttpClientFactory clientFactory)
        {
            _webHostEnvironment = webHostEnvironment;
            _clientFactory = clientFactory;
        }

        public IActionResult UserDashboard()
        {
            ViewBag.Message = "Model loaded successfully. You can now analyze X-ray images.";
            ViewBag.MessageType = "success";
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> UploadXRay(IFormFile xrayImage, string patientName, int? patientAge, string patientGender)
        {
            if (xrayImage == null || xrayImage.Length == 0)
            {
                ViewBag.Message = "Please select an image to analyze.";
                ViewBag.MessageType = "error";
                return View("UserDashboard");
            }

            if (string.IsNullOrEmpty(patientName))
            {
                ViewBag.Message = "Patient name is required.";
                ViewBag.MessageType = "error";
                return View("UserDashboard");
            }

            try
            {
                var userId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
                if (userId == 0)
                {
                    ViewBag.Message = "User not authenticated.";
                    ViewBag.MessageType = "error";
                    return View("UserDashboard");
                }

                var client = _clientFactory.CreateClient();
                client.Timeout = TimeSpan.FromSeconds(300); // 5 minutes timeout

                using var content = new MultipartFormDataContent();
                using var stream = xrayImage.OpenReadStream();
                content.Add(new StreamContent(stream), "file", xrayImage.FileName);
                content.Add(new StringContent(userId.ToString()), "user_id");
                content.Add(new StringContent(patientName), "patient_name");
                if (patientAge.HasValue)
                    content.Add(new StringContent(patientAge.Value.ToString()), "patient_age");
                if (!string.IsNullOrEmpty(patientGender))
                    content.Add(new StringContent(patientGender), "patient_gender");

                var response = await client.PostAsync("http://127.0.0.1:8000/predict", content);

                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var result = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
                    var pdfUrl = result["pdf_url"];
                    ViewBag.Message = "Analysis completed successfully!";
                    ViewBag.MessageType = "success";
                    ViewBag.PdfUrl = $"http://127.0.0.1:8000/{pdfUrl}";
                    return View("UserDashboard");
                }
                else
                {
                    var error = await response.Content.ReadAsStringAsync();
                    ViewBag.Message = $"Error from API: {response.StatusCode} - {error}";
                    ViewBag.MessageType = "error";
                    return View("UserDashboard");
                }
            }
            catch (TaskCanceledException ex)
            {
                ViewBag.Message = $"Request timed out: {ex.Message}. Ensure the FastAPI server is running on http://127.0.0.1:8000.";
                ViewBag.MessageType = "error";
                return View("UserDashboard");
            }
            catch (HttpRequestException ex)
            {
                ViewBag.Message = $"Network error: {ex.Message}. Check if FastAPI server is running on http://127.0.0.1:8000.";
                ViewBag.MessageType = "error";
                return View("UserDashboard");
            }
            catch (Exception ex)
            {
                ViewBag.Message = $"Error analyzing image: {ex.Message}";
                ViewBag.MessageType = "error";
                return View("UserDashboard");
            }
        }

        [HttpGet]
        public async Task<IActionResult> ViewReports()
        {
            try
            {
                var userId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
                if (userId == 0)
                {
                    ViewBag.Message = "User not authenticated.";
                    ViewBag.MessageType = "error";
                    return View("UserDashboard");
                }

                var client = _clientFactory.CreateClient();
                client.Timeout = TimeSpan.FromSeconds(300); // 5 minutes timeout
                var response = await client.GetAsync($"http://127.0.0.1:8000/user_pdfs/{userId}");

                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var pdfsData = JsonSerializer.Deserialize<Dictionary<string, List<Dictionary<string, object>>>>(json);
                    ViewBag.Pdfs = pdfsData["pdfs"];
                    return View();
                }
                else
                {
                    var error = await response.Content.ReadAsStringAsync();
                    ViewBag.Message = $"Error retrieving reports: {response.StatusCode} - {error}";
                    ViewBag.MessageType = "error";
                    return View("UserDashboard");
                }
            }
            catch (Exception ex)
            {
                ViewBag.Message = $"Error: {ex.Message}";
                ViewBag.MessageType = "error";
                return View("UserDashboard");
            }
        }
    }
}