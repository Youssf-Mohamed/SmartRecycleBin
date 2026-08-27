using Supabase.Postgrest.Attributes;
using Supabase.Postgrest.Models;

namespace SmartRecycleBin.Models
{
    [Table("predictions")]
    public class Prediction : BaseModel
    {
        [PrimaryKey("id", false)]
        public Guid Id { get; set; }

        [Column("user_id")]
        public Guid? UserId { get; set; }

        [Column("image_path")]
        public string ImagePath { get; set; } = string.Empty;

        [Column("predicted_label")]
        public string PredictedLabel { get; set; } = string.Empty;

        [Column("confidence_score")]
        public float ConfidenceScore { get; set; }

        [Column("all_scores")]
        public string AllScores { get; set; } = string.Empty;

        [Column("created_at")]
        public DateTime CreatedAt { get; set; }
    }
}
