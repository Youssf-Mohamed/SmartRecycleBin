using SmartRecycleBin.Models;
using Supabase;
using Supabase.Gotrue;
using Supabase.Postgrest;

namespace SmartRecycleBin.Services
{
    public class SupabaseService
    {
        private readonly Supabase.Client _client;

        public SupabaseService(IConfiguration configuration)
        {
            var url = configuration["Supabase:Url"]!;
            var key = configuration["Supabase:Key"]!;

            _client = new Supabase.Client(url, key);
            _client.InitializeAsync().GetAwaiter().GetResult();
        }

        public Supabase.Client Client => _client;

        public async Task<string> UploadImageAsync(byte[] imageBytes, string fileName)
        {
            var bucket = _client.Storage.From("images");
            var result = await bucket.Upload(imageBytes, $"predictions/{fileName}");

            if (result == null)
                throw new Exception("Failed to upload image to Supabase Storage");

            return $"predictions/{fileName}";
        }

        public async Task<Prediction> SavePredictionAsync(Prediction prediction)
        {
            var response = await _client
                .From<Prediction>()
                .Insert(prediction);

            return response.Model!;
        }

        public async Task<List<Prediction>> GetUserPredictionsAsync(Guid userId, int limit = 50)
        {
            var response = await _client
                .From<Prediction>()
                .Where(x => x.UserId == userId)
                .Order("created_at", Supabase.Postgrest.Constants.Ordering.Descending)
                .Limit(limit)
                .Get();

            return response.Models;
        }

        public async Task<string> GetSignedUrlAsync(string filePath, int expiresIn = 3600)
        {
            var bucket = _client.Storage.From("images");
            var url = await bucket.CreateSignedUrl(filePath, expiresIn);
            return url;
        }

        public User? GetCurrentUser()
        {
            return _client.Auth.CurrentUser;
        }

        public async Task<Session?> SignInAsync(string email, string password)
        {
            return await _client.Auth.SignIn(email, password);
        }

        public async Task<Session?> SignUpAsync(string email, string password)
        {
            return await _client.Auth.SignUp(email, password);
        }

        public async Task SignOutAsync()
        {
            await _client.Auth.SignOut();
        }
    }
}
