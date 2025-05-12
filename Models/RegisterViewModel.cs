using System.ComponentModel.DataAnnotations;

namespace WebApplication1.Models
{
    public class RegisterViewModel
    {

        
        [Required(ErrorMessage = "Email is required")]
        [EmailAddress(ErrorMessage = "Invalid email address")]
        public string Email { get; set; }

        [Required(ErrorMessage = "Password is required")]
        [DataType(DataType.Password)]
        [StringLength(100, ErrorMessage = "The password must be at least {2} characters long.", MinimumLength = 6)]
        public string Password { get; set; }
        
       [Display(Name = "Remember me")]
        public bool RememberMe { get; set; }=false;
        
       
    }
}