using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using SmartRecycleBin.Models;
using SmartRecycleBin.Services;

namespace SmartRecycleBin.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ClassificationController : ControllerBase
    {
        private readonly PredictionEnginePool<MLModel.ModelInput, MLModel.ModelOutput> _predictionEngine;
        private readonly SupabaseService _supabase;

        public ClassificationController(
            PredictionEnginePool<MLModel.ModelInput, MLModel.ModelOutput> predictionEngine,
            SupabaseService supabase)
        {
            _predictionEngine = predictionEngine;
            _supabase = supabase;
        }

        [HttpGet("health")]
        public IActionResult Health()
        {
            return Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
        }

        [HttpPost]
        public async Task<IActionResult> Classify(IFormFile file)
        {
            if (file == null || file.Length == 0)
                return BadRequest("No image file provided.");

            using var stream = new MemoryStream();
            await file.CopyToAsync(stream);
            var imageBytes = stream.ToArray();

            var input = new MLModel.ModelInput
            {
                ImageSource = imageBytes
            };

            var result = _predictionEngine.Predict(input);
            var scores = MLModel.GetSortedScoresWithLabels(result);

            var response = new ClassificationResponse
            {
                PredictedLabel = result.PredictedLabel,
                ConfidenceScore = result.Score?.Max() ?? 0f,
                AllScores = scores.ToDictionary(x => x.Key, x => x.Value)
            };

            return Ok(response);
        }

        [HttpPost("classify-and-save")]
        [Authorize]
        public async Task<IActionResult> ClassifyAndSave(IFormFile file)
        {
            if (file == null || file.Length == 0)
                return BadRequest("No image file provided.");

            var user = _supabase.GetCurrentUser();
            if (user == null)
                return Unauthorized();

            using var stream = new MemoryStream();
            await file.CopyToAsync(stream);
            var imageBytes = stream.ToArray();

            var input = new MLModel.ModelInput
            {
                ImageSource = imageBytes
            };

            var result = _predictionEngine.Predict(input);
            var scores = MLModel.GetSortedScoresWithLabels(result);

            var fileName = $"{Guid.NewGuid()}_{file.FileName}";
            var imagePath = await _supabase.UploadImageAsync(imageBytes, fileName);

            var prediction = new Prediction
            {
                Id = Guid.NewGuid(),
                UserId = Guid.Parse(user.Id),
                ImagePath = imagePath,
                PredictedLabel = result.PredictedLabel,
                ConfidenceScore = result.Score?.Max() ?? 0f,
                AllScores = System.Text.Json.JsonSerializer.Serialize(
                    scores.ToDictionary(x => x.Key, x => x.Value)),
                CreatedAt = DateTime.UtcNow
            };

            var saved = await _supabase.SavePredictionAsync(prediction);

            return Ok(new ClassificationResponse
            {
                Id = saved.Id,
                PredictedLabel = result.PredictedLabel,
                ConfidenceScore = result.Score?.Max() ?? 0f,
                AllScores = scores.ToDictionary(x => x.Key, x => x.Value),
                ImagePath = imagePath,
                CreatedAt = saved.CreatedAt
            });
        }

        [HttpGet("history")]
        [Authorize]
        public async Task<IActionResult> GetHistory([FromQuery] int limit = 50)
        {
            var user = _supabase.GetCurrentUser();
            if (user == null)
                return Unauthorized();

            var predictions = await _supabase.GetUserPredictionsAsync(Guid.Parse(user.Id), limit);
            return Ok(predictions);
        }

        [HttpPost("auth/signup")]
        public async Task<IActionResult> SignUp([FromBody] AuthRequest request)
        {
            try
            {
                var session = await _supabase.SignUpAsync(request.Email, request.Password);
                if (session == null)
                    return BadRequest("Sign up failed.");

                return Ok(new AuthResponse
                {
                    AccessToken = session.AccessToken,
                    RefreshToken = session.RefreshToken,
                    UserId = session.User?.Id
                });
            }
            catch (Exception ex)
            {
                return BadRequest(ex.Message);
            }
        }

        [HttpPost("auth/signin")]
        public async Task<IActionResult> SignIn([FromBody] AuthRequest request)
        {
            try
            {
                var session = await _supabase.SignInAsync(request.Email, request.Password);
                if (session == null)
                    return BadRequest("Sign in failed.");

                return Ok(new AuthResponse
                {
                    AccessToken = session.AccessToken,
                    RefreshToken = session.RefreshToken,
                    UserId = session.User?.Id
                });
            }
            catch (Exception ex)
            {
                return BadRequest(ex.Message);
            }
        }
    }

    public class ClassificationResponse
    {
        public Guid? Id { get; set; }
        public string PredictedLabel { get; set; } = string.Empty;
        public float ConfidenceScore { get; set; }
        public Dictionary<string, float> AllScores { get; set; } = new();
        public string? ImagePath { get; set; }
        public DateTime? CreatedAt { get; set; }
    }

    public class AuthRequest
    {
        public string Email { get; set; } = string.Empty;
        public string Password { get; set; } = string.Empty;
    }

    public class AuthResponse
    {
        public string? AccessToken { get; set; }
        public string? RefreshToken { get; set; }
        public string? UserId { get; set; }
    }
}
